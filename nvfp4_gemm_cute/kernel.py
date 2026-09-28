# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This file is ported from CUTLASS's sm103_dense_blockscaled_gemm_persistent.py
# with modifications for FlashInfer integration (alpha scaling, PDL support, wrapper method).
# Original: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py

from typing import Optional, Type, Tuple, Union
from dataclasses import dataclass, field
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm103_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from .epilogue import epilogue_with_alpha


class _PresetTileScheduler(utils.StaticPersistentTileScheduler):
    """Static scheduling with an M-band swizzle or the measured 8K ownership map."""

    fast_swz = 1
    owned_8k = False

    @staticmethod
    def create(params, block_idx, grid_dim, *, fast_swz=1, owned_8k=False, loc=None, ip=None):
        sched = utils.StaticPersistentTileScheduler.create(
            params, block_idx, grid_dim, loc=loc, ip=ip
        )
        cls = type(
            "_PresetSched_s%d_o%d" % (fast_swz, int(owned_8k)),
            (_PresetTileScheduler,),
            {"fast_swz": fast_swz, "owned_8k": owned_8k},
        )
        sched.__class__ = cls
        return sched

    def __new_from_mlir_values__(self, values):
        obj = super().__new_from_mlir_values__(values)
        obj.__class__ = self.__class__
        return obj

    def _get_current_work_for_linear_idx(self, current_work_linear_idx, *, loc=None, ip=None):
        # Exact nine-slot map; its two groups split the last N half evenly.
        if cutlass.const_expr(self.owned_8k):
            slot = cute.arch.block_idx()[2]
            wave = (current_work_linear_idx - slot) // 9
            in_a = cutlass.Int32(
                (slot == 0) | (slot == 1) | (slot == 4) | (slot == 5) | (slot == 8)
            )
            rank_a = (
                cutlass.Int32(slot == 1)
                + 2 * cutlass.Int32(slot == 4)
                + 3 * cutlass.Int32(slot == 5)
                + 4 * cutlass.Int32(slot == 8)
            )
            rank_b = (
                cutlass.Int32(slot == 3)
                + 2 * cutlass.Int32(slot == 6)
                + 3 * cutlass.Int32(slot == 7)
            )
            pa = rank_a + wave * 5
            pb = rank_b + wave * 4
            a_head = cutlass.Int32(pa < 40)
            b_head = cutlass.Int32(pb < 24)
            na = a_head * (pa // 5) + (1 - a_head) * (8 + (pa - 40) // 4)
            ma = a_head * (pa % 5) + (1 - a_head) * ((pa - 40) % 4)
            nb = b_head * (pb // 3) + (1 - b_head) * (8 + (pb - 24) // 4)
            mb = b_head * (5 + pb % 3) + (1 - b_head) * (4 + (pb - 24) % 4)
            mc = in_a * ma + (1 - in_a) * mb
            nc = in_a * na + (1 - in_a) * nb
            valid = (in_a != 0) & (pa < 72) | (in_a == 0) & (pb < 56)
            tile = (
                mc * 8 + self.cta_id_in_cluster[0],
                nc * 2 + self.cta_id_in_cluster[1],
                cutlass.Int32(0),
            )
            return utils.WorkTileInfo(tile, valid)
        # Low bits select M within a band; FastDivmod decodes the N sweep.
        if cutlass.const_expr(self.fast_swz > 1):
            s = self.fast_swz
            lg = s.bit_length() - 1
            lin = current_work_linear_idx
            m_in = lin & s - 1
            x = lin >> lg
            m_out, n = divmod(x, self.params.cluster_shape_major_fdd)
            ncl = cute.size(self.params.problem_layout_ncluster_mnl, mode=[1])
            j = (m_out * s + m_in) * ncl + n
            return super()._get_current_work_for_linear_idx(j, loc=loc, ip=ip)
        return super()._get_current_work_for_linear_idx(current_work_linear_idx, loc=loc, ip=ip)


class Sm103BlockScaledPersistentDenseGemmKernel:
    """Persistent NVFP4 GEMM for SM103. See docs/kernel.md for the pipeline and presets."""

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        enable_pdl: bool = True,
        fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None,
    ):
        """Configure the MMA tile, preferred/fallback clusters, and PDL."""
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        self.fallback_cluster_shape_mn = fallback_cluster_shape_mn
        self.is_mixed_cluster = fallback_cluster_shape_mn is not None and tuple(
            fallback_cluster_shape_mn
        ) != tuple(cluster_shape_mn)
        if self.is_mixed_cluster:
            from .hw import get_max_active_clusters

            fb_size = fallback_cluster_shape_mn[0] * fallback_cluster_shape_mn[1]
            self.mixed_num_slots = (
                get_max_active_clusters(fb_size)
                * fb_size
                // (cluster_shape_mn[0] * cluster_shape_mn[1])
            )
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.enable_pdl = enable_pdl
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_ab_warp_id = 5
        self.tma_sf_warp_id = 6
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_ab_warp_id, self.tma_sf_warp_id, *self.epilogue_warp_id)
        )
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_103")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_103")

    def _setup_attributes(self):
        """Set up kernel attributes that depend on runtime tensor inputs.

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        """
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )
        tiled_mma = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype, self.cta_group, self.mma_inst_shape_mn
        )
        dummy_tiled_mma_sfb = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype, tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb
        )
        self.mma_tiler = (self.mma_inst_shape_mn[0], self.mma_inst_shape_mn[1], 768)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_layout_vmnk.shape[0]),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_n_sf = cute.round_up(cute.size(self.cta_tile_shape_mnk[1]), 128)
        self.mma_sf_tiler = (
            self.cta_tile_shape_mnk[0],
            self.cta_n_sf,
            self.cta_tile_shape_mnk[2] // 4,
        )
        self.sf_atom = self.Sm103BlockScaledBasicChunk(16, tiled_mma.op.a_major_mode).layout
        shapes = [tuple(self.cluster_shape_mn)]
        if self.is_mixed_cluster:
            shapes.append(tuple(self.fallback_cluster_shape_mn))
        self._cluster_cfgs = []
        for shape in shapes:
            layout_vmnk = cute.tiled_divide(
                cute.make_layout((*shape, 1)), (tiled_mma.thr_id.shape,)
            )
            layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout((*shape, 1)), (dummy_tiled_mma_sfb.thr_id.shape,)
            )
            self._cluster_cfgs.append(
                dict(
                    cluster_shape_mn=shape,
                    cluster_layout_vmnk=layout_vmnk,
                    cluster_layout_sfb_vmnk=layout_sfb_vmnk,
                    num_mcast_ctas_a=cute.size(layout_vmnk.shape[2]),
                    num_mcast_ctas_b=cute.size(layout_vmnk.shape[1]),
                    num_mcast_ctas_sfb=cute.size(layout_sfb_vmnk.shape[1]),
                )
            )
        self._use_cluster_cfg(0)
        self.epi_tile = sm103_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype
        )
        self.num_acc_stage, self.num_ab_stage, self.num_sf_stage = self._compute_stages(
            tiled_mma, self.mma_tiler, self.sf_dtype, self.smem_capacity, self.occupancy
        )
        self.a_smem_layout_staged = self.sm103_make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.num_ab_stage
        )
        self.a_smem_layout_staged_tma = self.sm103_make_smem_layout_a(tiled_mma, self.mma_tiler, 3)
        self.b_smem_layout_staged = self.sm103_make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.num_ab_stage
        )
        self.b_smem_layout_staged_tma = self.sm103_make_smem_layout_b(tiled_mma, self.mma_tiler, 3)
        self.sfa_smem_layout_staged = self.sm103_make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.num_sf_stage
        )
        self.sfb_smem_layout_staged = self.sm103_make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.num_sf_stage
        )
        self.overlapping_accum = self.num_acc_stage == 1
        self.epi_tile_n = cute.size(self.epi_tile[1])
        if self.overlapping_accum:

            def _sf_tmem_cols(make_tmem_layout_fn, smem_layout_staged):
                layout = make_tmem_layout_fn(
                    tiled_mma,
                    self.mma_tiler,
                    16,
                    cute.slice_(smem_layout_staged, (None, None, None, 0)),
                )
                return cute.cosize(cute.recast_layout(32, self.sf_dtype.width, layout)) & 65535

            self.num_sfa_tmem_cols = _sf_tmem_cols(
                blockscaled_utils.make_tmem_layout_sfa, self.sfa_smem_layout_staged
            )
            self.num_sfb_tmem_cols = _sf_tmem_cols(
                blockscaled_utils.make_tmem_layout_sfb, self.sfb_smem_layout_staged
            )
            self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
            self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n

    packed_ab_desc = False
    specialize_32k = False
    trim_short_tail = False
    retire_a_cols = 0
    tma_b_first = False
    tma_sfb_first = False
    epi_store_evict_first_min_k_bytes = 4096
    epi_store_pace_per_ktile_ns = 55
    raster_along_m = True
    fast_swz = 1
    _L2_POLICIES = {
        "evict_normal": 1152921504606846976,
        "evict_first": 1364590687093260288,
        "evict_last": 1508705875169116160,
    }
    l2_policy_a = None
    l2_policy_b = None

    def _tma_load_cache_policy(self, kind: str):
        """EVICT_LAST for operand loads when enabled, so the output stream (larger
        than L2) evicts output lines instead of A/B/SF that are still being reused."""
        pol = getattr(self, f"l2_policy_{kind}", None)
        if pol is None and kind in ("sfa", "sfb"):
            pol = None
        if pol is not None:
            return cutlass.Int64(self._L2_POLICIES[pol])
        return None

    def _make_tile_sched(self, tile_sched_params):
        fswz = getattr(self, "fast_swz", 1)
        owned_8k = getattr(self, "owned_8k", False)
        if fswz > 1 or owned_8k:
            return _PresetTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                fast_swz=fswz,
                owned_8k=owned_8k,
            )
        return utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )

    def _use_cluster_cfg(self, idx: int) -> None:
        """Select the cluster shape that host setup / kernel tracing specializes on."""
        for key, value in self._cluster_cfgs[idx].items():
            setattr(self, key, value)
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param alpha: Single-element tensor containing alpha scaling value
        :type alpha: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        self._setup_attributes()
        sfa_layout = cute.tile_to_shape(self.sf_atom, a_tensor.shape, (2, 1, 3))
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        sfb_layout = cute.tile_to_shape(self.sf_atom, b_tensor.shape, (2, 1, 3))
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)
        tiled_mma = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype, self.cta_group, self.mma_inst_shape_mn
        )
        dummy_tiled_mma_sfb = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype, tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        tma_atoms_a, tma_tensors_a, tma_atoms_b, tma_tensors_b = ([], [], [], [])
        tma_atoms_sfa, tma_tensors_sfa, tma_atoms_sfb, tma_tensors_sfb = ([], [], [], [])
        for cfg_idx in cutlass.range_constexpr(len(self._cluster_cfgs)):
            self._use_cluster_cfg(cfg_idx)
            a_op = sm103_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
            a_smem_layout_tma_ready = self.adapt_layout_for_tma_ab(self.a_smem_layout_staged_tma)
            a_tensor_uint8 = cute.recast_tensor(a_tensor, cutlass.Uint8)
            tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
                a_op,
                a_tensor_uint8,
                a_smem_layout_tma_ready,
                (cute.size(tiled_mma.tv_layout_A[1][0]), 384),
                self.cluster_shape_mn[1],
                internal_type=cutlass.Uint8,
            )
            b_op = sm103_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
            b_smem_layout_tma_ready = self.adapt_layout_for_tma_ab(self.b_smem_layout_staged_tma)
            b_tensor_uint8 = cute.recast_tensor(b_tensor, cutlass.Uint8)
            tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
                b_op,
                b_tensor_uint8,
                b_smem_layout_tma_ready,
                (cute.size(tiled_mma.tv_layout_B[1][0]), 384),
                self.cluster_shape_mn[0] // cute.size(tiled_mma.thr_id.shape),
                internal_type=cutlass.Uint8,
            )
            sfa_op = sm103_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
            sfa_smem_layout_tma_ready = self.adapt_layout_for_tma_sf(sfa_smem_layout)
            tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.cpasync.make_tiled_tma_atom(
                sfa_op,
                sfa_tensor,
                sfa_smem_layout_tma_ready,
                (self.mma_sf_tiler[0], self.mma_sf_tiler[2]),
                self.cluster_shape_mn[1],
                internal_type=cutlass.Uint8,
            )
            sfb_op = sm103_utils.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
            sfb_smem_layout_tma_ready = self.adapt_layout_for_tma_sf(sfb_smem_layout)
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.cpasync.make_tiled_tma_atom(
                sfb_op,
                sfb_tensor,
                sfb_smem_layout_tma_ready,
                (self.mma_sf_tiler[1], self.mma_sf_tiler[2]),
                self.cluster_shape_mn[0] // cute.size(dummy_tiled_mma_sfb.thr_id),
                internal_type=cutlass.Uint8,
            )
            tma_atoms_a.append(tma_atom_a)
            tma_tensors_a.append(tma_tensor_a)
            tma_atoms_b.append(tma_atom_b)
            tma_tensors_b.append(tma_tensor_b)
            tma_atoms_sfa.append(tma_atom_sfa)
            tma_tensors_sfa.append(tma_tensor_sfa)
            tma_atoms_sfb.append(tma_atom_sfb)
            tma_tensors_sfb.append(tma_tensor_sfb)
        self._use_cluster_cfg(0)
        a_copy_size = cute.size_in_bytes(
            cutlass.Uint8, cute.slice_(self.a_smem_layout_staged_tma, (None, None, None, 0))
        )
        b_copy_size = cute.size_in_bytes(
            cutlass.Uint8, cute.slice_(self.b_smem_layout_staged_tma, (None, None, None, 0))
        )
        sfa_copy_size = cute.size_in_bytes(
            cutlass.Uint8, cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        )
        sfb_copy_size = cute.size_in_bytes(
            cutlass.Uint8, cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        )
        self.num_tma_load_bytes_ab = (a_copy_size + b_copy_size) * atom_thr_size
        self.num_tma_load_bytes_sf = (sfa_copy_size + sfb_copy_size) * atom_thr_size
        if cutlass.const_expr(self.is_mixed_cluster):
            max_active_clusters = self.mixed_num_slots
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            1,
            self.raster_along_m,
        )
        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            sf_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_sf_stage]
            sf_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_sf_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tuple(tma_atoms_a),
            tuple(tma_tensors_a),
            tuple(tma_atoms_b),
            tuple(tma_tensors_b),
            tuple(tma_atoms_sfa),
            tuple(tma_tensors_sfa),
            tuple(tma_atoms_sfb),
            tuple(tma_tensors_sfb),
            c_tensor,
            tuple((cfg["cluster_layout_vmnk"] for cfg in self._cluster_cfgs)),
            tuple((cfg["cluster_layout_sfb_vmnk"] for cfg in self._cluster_cfgs)),
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            alpha,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1)
            if self.is_mixed_cluster
            else None,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atoms_a: Tuple[cute.CopyAtom, ...],
        mA_mkls: Tuple[cute.Tensor, ...],
        tma_atoms_b: Tuple[cute.CopyAtom, ...],
        mB_nkls: Tuple[cute.Tensor, ...],
        tma_atoms_sfa: Tuple[cute.CopyAtom, ...],
        mSFA_mkls: Tuple[cute.Tensor, ...],
        tma_atoms_sfb: Tuple[cute.CopyAtom, ...],
        mSFB_nkls: Tuple[cute.Tensor, ...],
        mC_mnl: cute.Tensor,
        cluster_layouts_vmnk: Tuple[cute.Layout, ...],
        cluster_layouts_sfb_vmnk: Tuple[cute.Layout, ...],
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.

        Mixed-cluster launches trace ``kernel_body`` once per cluster shape and pick
        the branch from the runtime cluster dims. Tile ownership comes from the
        preferred-shape scheduler in both branches (a fallback cluster computes the
        tiles its CTAs would own inside a preferred cluster), so the branches differ
        only in multicast, never in which tiles they write.
        """
        # Allocate before dispatch so both physical cluster branches share storage.
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # Multicast follows the physical cluster; ownership follows the preferred one.
        if cutlass.const_expr(self.is_mixed_cluster):
            cdim_x, cdim_y, _ = cute.arch.block_in_cluster_dim()
            if cdim_x == self.cluster_shape_mn[0] and cdim_y == self.cluster_shape_mn[1]:
                self.kernel_body(
                    0,
                    storage,
                    tiled_mma,
                    tma_atoms_a[0],
                    mA_mkls[0],
                    tma_atoms_b[0],
                    mB_nkls[0],
                    tma_atoms_sfa[0],
                    mSFA_mkls[0],
                    tma_atoms_sfb[0],
                    mSFB_nkls[0],
                    mC_mnl,
                    cluster_layouts_vmnk[0],
                    cluster_layouts_sfb_vmnk[0],
                    a_smem_layout_staged,
                    b_smem_layout_staged,
                    sfa_smem_layout_staged,
                    sfb_smem_layout_staged,
                    epi_tile,
                    tile_sched_params,
                    epilogue_op,
                    alpha,
                )
            else:
                self.kernel_body(
                    1,
                    storage,
                    tiled_mma,
                    tma_atoms_a[1],
                    mA_mkls[1],
                    tma_atoms_b[1],
                    mB_nkls[1],
                    tma_atoms_sfa[1],
                    mSFA_mkls[1],
                    tma_atoms_sfb[1],
                    mSFB_nkls[1],
                    mC_mnl,
                    cluster_layouts_vmnk[1],
                    cluster_layouts_sfb_vmnk[1],
                    a_smem_layout_staged,
                    b_smem_layout_staged,
                    sfa_smem_layout_staged,
                    sfb_smem_layout_staged,
                    epi_tile,
                    tile_sched_params,
                    epilogue_op,
                    alpha,
                )
        else:
            self.kernel_body(
                0,
                storage,
                tiled_mma,
                tma_atoms_a[0],
                mA_mkls[0],
                tma_atoms_b[0],
                mB_nkls[0],
                tma_atoms_sfa[0],
                mSFA_mkls[0],
                tma_atoms_sfb[0],
                mSFB_nkls[0],
                mC_mnl,
                cluster_layouts_vmnk[0],
                cluster_layouts_sfb_vmnk[0],
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                epi_tile,
                tile_sched_params,
                epilogue_op,
                alpha,
            )

    @cute.jit
    def kernel_body(
        self,
        cluster_cfg_idx: cutlass.Constexpr,
        storage,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        """Kernel body, specialized on cluster shape ``_cluster_cfgs[cluster_cfg_idx]``."""
        self._use_cluster_cfg(cluster_cfg_idx)
        alpha_value = alpha[0].to(cutlass.Float32)
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == self.tma_ab_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        if warp_idx == self.tma_sf_warp_id:
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()
        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.num_tma_load_bytes_ab,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        sf_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_sf_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        sf_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_sf_tma_producer)
        sf_producer, sf_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.sf_full_mbar_ptr.data_ptr(),
            num_stages=self.num_sf_stage,
            producer_group=sf_producer_group,
            consumer_group=sf_consumer_group,
            tx_count=self.num_tma_load_bytes_sf,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = None
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_dealloc_sync_bar_id, num_threads=32 * len(self.epilogue_warp_id)
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_((self.mma_tiler[0], self.mma_tiler[1], 384), (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_((self.mma_tiler[0], self.mma_tiler[1], 384), (0, None, None)),
            (None, None, None),
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_sf_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_sf_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        # A K=256 tail still needs the zero-filled second AB stage for MMA 2.
        short_k_tail = cute.size(mA_mkl, mode=[1]) - (k_tile_cnt - 1) * 384 <= 192
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA_mkl_tmp = thr_mma.partition_A(gA_mkl)
        tCgA_layout = self.append_coalesce_layout(tCgA_mkl_tmp.layout)
        cta_tCgA = cute.make_tensor(tCgA_mkl_tmp.iterator, tCgA_layout)
        tCgA = cute.make_tensor(
            cta_tCgA.iterator,
            cute.tiled_divide(cta_tCgA.layout, (cute.size(tiled_mma.tv_layout_A[1][0]), 128)),
        )
        tCgB_nkl_tmp = thr_mma.partition_B(gB_nkl)
        tCgB_layout = self.append_coalesce_layout(tCgB_nkl_tmp.layout)
        cta_tCgB = cute.make_tensor(tCgB_nkl_tmp.iterator, tCgB_layout)
        tCgB = cute.make_tensor(
            cta_tCgB.iterator,
            cute.tiled_divide(cta_tCgB.layout, (cute.size(tiled_mma.tv_layout_B[1][0]), 128)),
        )
        tCgSFA = cute.make_tensor(
            gSFA_mkl.iterator,
            cute.tiled_divide(gSFA_mkl.layout, (self.mma_sf_tiler[0], self.mma_sf_tiler[2])),
        )
        tCgSFB = cute.make_tensor(
            gSFB_nkl.iterator,
            cute.tiled_divide(gSFB_nkl.layout, (self.mma_sf_tiler[1], self.mma_sf_tiler[2])),
        )
        tCgC = thr_mma.partition_C(gC_mnl)
        idC = cute.make_identity_tensor(mC_mnl.shape)
        cC_mnl = cute.local_tile(
            idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        tCcC = thr_mma.partition_C(cC_mnl)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 1),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 1),
        )
        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA_compact = cute.filter_zeros(tAsSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB_compact = cute.filter_zeros(tBsSFB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (self.cta_tile_shape_mnk[1] - self.num_sf_tmem_cols)
                        * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        if cutlass.const_expr(getattr(self, "prefetch_first_ktiles", 0) > 0):
            if warp_idx == self.tma_ab_warp_id or warp_idx == self.tma_sf_warp_id:
                pf_sched = self._make_tile_sched(tile_sched_params)
                pf_first = pf_sched.initial_work_tile_info()
                if pf_first.is_valid_tile:
                    pf_c = pf_first.tile_idx
                    n_pf = min(self.prefetch_first_ktiles, 16)
                    if warp_idx == self.tma_ab_warp_id:
                        pf_a = tAgA[
                            None,
                            None,
                            None,
                            pf_c[0] // cute.size(tiled_mma.thr_id.shape),
                            None,
                            pf_c[2],
                        ]
                        pf_b = tBgB[None, None, None, pf_c[1], None, pf_c[2]]
                        for pk in cutlass.range_constexpr(n_pf):
                            if pk < k_tile_cnt:
                                for pbuf in cutlass.range_constexpr(3):
                                    cute.prefetch(
                                        tma_atom_a,
                                        cute.group_modes(pf_a[None, None, pbuf, pk], 0, 2),
                                    )
                                    cute.prefetch(
                                        tma_atom_b,
                                        cute.group_modes(pf_b[None, None, pbuf, pk], 0, 2),
                                    )
                    else:
                        pf_sfa = tAgSFA[None, pf_c[0], None, pf_c[2]]
                        pf_sfb = tBgSFB[None, pf_c[1], None, pf_c[2]]
                        for pk in cutlass.range_constexpr(n_pf):
                            if pk < k_tile_cnt:
                                for pst in cutlass.range_constexpr(4):
                                    pidx = pk * 4 + pst
                                    cute.prefetch(
                                        tma_atom_sfa, cute.filter_zeros(pf_sfa[None, pidx])
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb, cute.filter_zeros(pf_sfb[None, pidx])
                                    )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        griddepcontrol_wait()
        tile_sched = self._make_tile_sched(tile_sched_params)
        work_tile = tile_sched.initial_work_tile_info()
        if warp_idx == self.tma_ab_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                a_cache_policy = self._tma_load_cache_policy("a")
                if cutlass.const_expr(self.retire_a_cols > 0):
                    a_cache_policy = cutlass.Int64(self._L2_POLICIES["evict_last"])
                    if (
                        mma_tile_coord_mnl[1]
                        >= cute.ceil_div(cute.size(mB_nkl, mode=[0]), self.mma_tiler[1])
                        - self.retire_a_cols
                    ):
                        a_cache_policy = cutlass.Int64(self._L2_POLICIES["evict_first"])
                tAgA_slice = tAgA[
                    None, None, None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]
                ]
                tBgB_slice = tBgB[
                    None, None, None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]
                ]
                ab_producer.reset()
                peek_ab_empty_status = cutlass.Boolean(1)
                peek_ab_empty_status = ab_producer.try_acquire()
                for k_tile in cutlass.range(
                    0, k_tile_cnt, 1, unroll=2 if self.specialize_32k else 1
                ):
                    for buffer in cutlass.range(3, unroll_full=True):
                        if cutlass.const_expr(self.trim_short_tail):
                            # Producer and consumer omit exactly the same tail stage.
                            live_ab = not (
                                k_tile == k_tile_cnt - 1 and short_k_tail and (buffer == 2)
                            )
                        else:
                            live_ab = True
                        if live_ab:
                            ab_empty = ab_producer.acquire_and_advance(peek_ab_empty_status)
                            for op_idx in cutlass.range_constexpr(2):
                                if cutlass.const_expr((op_idx == 0) == self.tma_b_first):
                                    cute.copy(
                                        tma_atom_b,
                                        cute.group_modes(
                                            tBgB_slice[None, None, buffer, k_tile], 0, 2
                                        ),
                                        tBsB[None, ab_empty.index],
                                        tma_bar_ptr=ab_empty.barrier,
                                        mcast_mask=b_full_mcast_mask,
                                        cache_policy=self._tma_load_cache_policy("b"),
                                    )
                                else:
                                    cute.copy(
                                        tma_atom_a,
                                        cute.group_modes(
                                            tAgA_slice[None, None, buffer, k_tile], 0, 2
                                        ),
                                        tAsA[None, ab_empty.index],
                                        tma_bar_ptr=ab_empty.barrier,
                                        mcast_mask=a_full_mcast_mask,
                                        cache_policy=a_cache_policy,
                                    )
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if not (k_tile == k_tile_cnt - 1 and buffer == 2):
                                peek_ab_empty_status = ab_producer.try_acquire()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_producer.tail()
        if warp_idx == self.tma_sf_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (cur_tile_coord[0], cur_tile_coord[1], cur_tile_coord[2])
                a_cache_policy = self._tma_load_cache_policy("sfa")
                if cutlass.const_expr(self.retire_a_cols > 0):
                    a_cache_policy = cutlass.Int64(self._L2_POLICIES["evict_last"])
                    if (
                        mma_tile_coord_mnl[1]
                        >= cute.ceil_div(cute.size(mB_nkl, mode=[0]), self.mma_tiler[1])
                        - self.retire_a_cols
                    ):
                        a_cache_policy = cutlass.Int64(self._L2_POLICIES["evict_first"])
                tAgSFA_slice = tAgSFA[None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]]
                tBgSFB_slice = tBgSFB[None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]]
                sf_producer.reset()
                peek_sf_empty_status = cutlass.Boolean(1)
                peek_sf_empty_status = sf_producer.try_acquire()
                for k_tile in cutlass.range(
                    0, k_tile_cnt, 1, unroll=2 if self.specialize_32k else 1
                ):
                    for sf_stage in cutlass.range(4, unroll_full=True):
                        if cutlass.const_expr(self.trim_short_tail):
                            live_sf = not (
                                k_tile == k_tile_cnt - 1 and short_k_tail and (sf_stage >= 2)
                            )
                        else:
                            live_sf = True
                        if live_sf:
                            sf_empty = sf_producer.acquire_and_advance(peek_sf_empty_status)
                            tAgSFA_compact = cute.filter_zeros(
                                tAgSFA_slice[None, k_tile * 4 + sf_stage]
                            )
                            tBgSFB_compact = cute.filter_zeros(
                                tBgSFB_slice[None, k_tile * 4 + sf_stage]
                            )
                            for op_idx in cutlass.range_constexpr(2):
                                if cutlass.const_expr((op_idx == 0) == self.tma_sfb_first):
                                    cute.copy(
                                        tma_atom_sfb,
                                        tBgSFB_compact,
                                        tBsSFB_compact[None, sf_empty.index],
                                        tma_bar_ptr=sf_empty.barrier,
                                        mcast_mask=sfb_full_mcast_mask,
                                        cache_policy=self._tma_load_cache_policy("sfb"),
                                    )
                                else:
                                    cute.copy(
                                        tma_atom_sfa,
                                        tAgSFA_compact,
                                        tAsSFA_compact[None, sf_empty.index],
                                        tma_bar_ptr=sf_empty.barrier,
                                        mcast_mask=sfa_full_mcast_mask,
                                        cache_policy=a_cache_policy,
                                    )
                            peek_sf_empty_status = cutlass.Boolean(1)
                            if not (k_tile == k_tile_cnt - 1 and sf_stage == 3):
                                peek_sf_empty_status = sf_producer.try_acquire()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            sf_producer.tail()
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base), dtype=self.sf_dtype
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                16,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            MMA_M = self.cta_tile_shape_mnk[0]
            MMA_N_SF = self.cta_n_sf
            MMA_K_SF = self.cta_tile_shape_mnk[2] // 2
            mma_iter_SFA_shape = (((32, 4), MMA_M // 128), (16, 1))
            sSFA_iter_shape = (mma_iter_SFA_shape, 1, MMA_K_SF // 16)
            sSFA_iter_layout = cute.make_layout(sSFA_iter_shape)
            mma_iter_SFB_shape = (((32, 4), MMA_N_SF // 128), (16, 1))
            sSFB_iter_shape = (mma_iter_SFB_shape, 1, MMA_K_SF // 16)
            sSFB_iter_layout = cute.make_layout(sSFB_iter_shape)
            tCtSFA_layout_mma = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, 16, sSFA_iter_layout
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            tCtSFA_mma = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout_mma)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                16,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB_layout_mma = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, 16, sSFB_iter_layout
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            tCtSFB_mma = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout_mma)
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            k_bytes = cute.size(mA_mkl, mode=[1])
            # The final MMA can cross K; TMA zero-fill remains necessary.
            tail_live_mmas = (k_bytes - (k_tile_cnt - 1) * 384 + 47) // 48
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[None, 0, 0, acc_stage_index]
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()
                sf_consumer.reset()
                peek_sf_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_sf_full_status = sf_consumer.try_wait()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                is_first_iteration = True
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    num_live_mmas = cutlass.Int32(8)
                    if k_tile == k_tile_cnt - 1:
                        num_live_mmas = tail_live_mmas
                    if is_leader_cta:
                        # Address preparation is independent of operand readiness.
                        if cutlass.const_expr(self.packed_ab_desc):
                            d0 = ab_consumer._PipelineConsumer__state.index
                            d1 = d0 + 1
                            if d1 == self.num_ab_stage:
                                d1 = 0
                            d2 = d1 + 1
                            if d2 == self.num_ab_stage:
                                d2 = 0
                            a0 = self._ab_descriptor(sA, d0, d0)
                            a01 = self._ab_descriptor(sA, d0, d1)
                            a1 = self._ab_descriptor(sA, d1, d1)
                            a12 = self._ab_descriptor(sA, d1, d2)
                            a2 = self._ab_descriptor(sA, d2, d2)
                            b0 = self._ab_descriptor(sB, d0, d0)
                            b01 = self._ab_descriptor(sB, d0, d1)
                            b1 = self._ab_descriptor(sB, d1, d1)
                            b12 = self._ab_descriptor(sB, d1, d2)
                            b2 = self._ab_descriptor(sB, d2, d2)
                        sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                        s2t_stage_coord = (None, None, None, None, sf_full.index)
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )
                        sf_full.release()
                        peek_sf_full_status = cutlass.Boolean(1)
                        peek_sf_full_status = sf_consumer.try_wait()
                        ab_full0 = ab_consumer.wait_and_advance(peek_ab_full_status)
                        peek_ab_full_status = cutlass.Boolean(1)
                        peek_ab_full_status = ab_consumer.try_wait()
                        if is_first_iteration:
                            acc_pipeline.producer_acquire(acc_producer_state)
                            is_first_iteration = False
                        k_block_coord_cur = (None, 0, 0, ab_full0.index)
                        k_block_coord_next = (None, 0, 0, ab_full0.index)
                        sf_kblock_coord = (None, None, 0)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                        if cutlass.const_expr(self.packed_ab_desc):
                            self._mma_prebuilt(tiled_mma, tCtAcc, a0, b0, 0)
                        else:
                            self.make_desc_and_call_mma(
                                tiled_mma,
                                tCtAcc,
                                sA[k_block_coord_cur],
                                sA[k_block_coord_next],
                                sB[k_block_coord_cur],
                                sB[k_block_coord_next],
                                tCtAcc,
                            )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        k_block_coord_cur = (None, 0, 3, ab_full0.index)
                        k_block_coord_next = (None, 0, 0, ab_full0.index)
                        sf_kblock_coord = (None, None, 6)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                        if num_live_mmas > 1:
                            if cutlass.const_expr(self.packed_ab_desc):
                                self._mma_prebuilt(tiled_mma, tCtAcc, a0, b0, 3)
                            else:
                                self.make_desc_and_call_mma(
                                    tiled_mma,
                                    tCtAcc,
                                    sA[k_block_coord_cur],
                                    sA[k_block_coord_next],
                                    sB[k_block_coord_cur],
                                    sB[k_block_coord_next],
                                    tCtAcc,
                                )
                        sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                        s2t_stage_coord = (None, None, None, None, sf_full.index)
                        if num_live_mmas > 2:
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                        sf_full.release()
                        peek_sf_full_status = cutlass.Boolean(1)
                        if cutlass.const_expr(self.trim_short_tail):
                            if num_live_mmas > 4:
                                peek_sf_full_status = sf_consumer.try_wait()
                        else:
                            peek_sf_full_status = sf_consumer.try_wait()
                        ab_full1 = ab_consumer.wait_and_advance(peek_ab_full_status)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if cutlass.const_expr(self.trim_short_tail):
                            if num_live_mmas > 4:
                                peek_ab_full_status = ab_consumer.try_wait()
                        else:
                            peek_ab_full_status = ab_consumer.try_wait()
                        k_block_coord_cur = (None, 0, 6, ab_full0.index)
                        k_block_coord_next = (None, 0, 0, ab_full1.index)
                        sf_kblock_coord = (None, None, 0)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                        if num_live_mmas > 2:
                            if cutlass.const_expr(self.packed_ab_desc):
                                self._mma_prebuilt(tiled_mma, tCtAcc, a01, b01, 6)
                            else:
                                self.make_desc_and_call_mma(
                                    tiled_mma,
                                    tCtAcc,
                                    sA[k_block_coord_cur],
                                    sA[k_block_coord_next],
                                    sB[k_block_coord_cur],
                                    sB[k_block_coord_next],
                                    tCtAcc,
                                )
                        ab_full0.release()
                        k_block_coord_cur = (None, 0, 1, ab_full1.index)
                        k_block_coord_next = (None, 0, 0, ab_full1.index)
                        sf_kblock_coord = (None, None, 6)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                        if num_live_mmas > 3:
                            if cutlass.const_expr(self.packed_ab_desc):
                                self._mma_prebuilt(tiled_mma, tCtAcc, a1, b1, 1)
                            else:
                                self.make_desc_and_call_mma(
                                    tiled_mma,
                                    tCtAcc,
                                    sA[k_block_coord_cur],
                                    sA[k_block_coord_next],
                                    sB[k_block_coord_cur],
                                    sB[k_block_coord_next],
                                    tCtAcc,
                                )
                        if cutlass.const_expr(self.trim_short_tail):
                            run_tail_half = num_live_mmas > 4
                        else:
                            run_tail_half = True
                        if run_tail_half:
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (None, None, None, None, sf_full.index)
                            if num_live_mmas > 4:
                                cute.copy(
                                    tiled_copy_s2t_sfa,
                                    tCsSFA_compact_s2t[s2t_stage_coord],
                                    tCtSFA_compact_s2t,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfb,
                                    tCsSFB_compact_s2t[s2t_stage_coord],
                                    tCtSFB_compact_s2t,
                                )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            peek_sf_full_status = sf_consumer.try_wait()
                            k_block_coord_cur = (None, 0, 4, ab_full1.index)
                            k_block_coord_next = (None, 0, 0, ab_full1.index)
                            sf_kblock_coord = (None, None, 0)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                            if num_live_mmas > 4:
                                if cutlass.const_expr(self.packed_ab_desc):
                                    self._mma_prebuilt(tiled_mma, tCtAcc, a1, b1, 4)
                                else:
                                    self.make_desc_and_call_mma(
                                        tiled_mma,
                                        tCtAcc,
                                        sA[k_block_coord_cur],
                                        sA[k_block_coord_next],
                                        sB[k_block_coord_cur],
                                        sB[k_block_coord_next],
                                        tCtAcc,
                                    )
                            ab_full2 = ab_consumer.wait_and_advance(peek_ab_full_status)
                            peek_ab_full_status = cutlass.Boolean(1)
                            if k_tile + 1 < k_tile_cnt:
                                peek_ab_full_status = ab_consumer.try_wait()
                            k_block_coord_cur = (None, 0, 7, ab_full1.index)
                            k_block_coord_next = (None, 0, 0, ab_full2.index)
                            sf_kblock_coord = (None, None, 6)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                            if num_live_mmas > 5:
                                if cutlass.const_expr(self.packed_ab_desc):
                                    self._mma_prebuilt(tiled_mma, tCtAcc, a12, b12, 7)
                                else:
                                    self.make_desc_and_call_mma(
                                        tiled_mma,
                                        tCtAcc,
                                        sA[k_block_coord_cur],
                                        sA[k_block_coord_next],
                                        sB[k_block_coord_cur],
                                        sB[k_block_coord_next],
                                        tCtAcc,
                                    )
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (None, None, None, None, sf_full.index)
                            if num_live_mmas > 6:
                                cute.copy(
                                    tiled_copy_s2t_sfa,
                                    tCsSFA_compact_s2t[s2t_stage_coord],
                                    tCtSFA_compact_s2t,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfb,
                                    tCsSFB_compact_s2t[s2t_stage_coord],
                                    tCtSFB_compact_s2t,
                                )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            if k_tile + 1 < k_tile_cnt:
                                peek_sf_full_status = sf_consumer.try_wait()
                            ab_full1.release()
                            k_block_coord_cur = (None, 0, 2, ab_full2.index)
                            k_block_coord_next = (None, 0, 0, ab_full2.index)
                            sf_kblock_coord = (None, None, 0)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                            if num_live_mmas > 6:
                                if cutlass.const_expr(self.packed_ab_desc):
                                    self._mma_prebuilt(tiled_mma, tCtAcc, a2, b2, 2)
                                else:
                                    self.make_desc_and_call_mma(
                                        tiled_mma,
                                        tCtAcc,
                                        sA[k_block_coord_cur],
                                        sA[k_block_coord_next],
                                        sB[k_block_coord_cur],
                                        sB[k_block_coord_next],
                                        tCtAcc,
                                    )
                            k_block_coord_cur = (None, 0, 5, ab_full2.index)
                            k_block_coord_next = (None, 0, 0, ab_full2.index)
                            sf_kblock_coord = (None, None, 6)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                            if num_live_mmas > 7:
                                if cutlass.const_expr(self.packed_ab_desc):
                                    self._mma_prebuilt(tiled_mma, tCtAcc, a2, b2, 5)
                                else:
                                    self.make_desc_and_call_mma(
                                        tiled_mma,
                                        tCtAcc,
                                        sA[k_block_coord_cur],
                                        sA[k_block_coord_next],
                                        sB[k_block_coord_cur],
                                        sB[k_block_coord_next],
                                        tCtAcc,
                                    )
                            ab_full2.release()
                        else:
                            ab_full1.release()
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                acc_consumer_state = epilogue_with_alpha(
                    self,
                    tidx,
                    tCtAcc_base,
                    tCgC,
                    epi_tile,
                    epilogue_op,
                    alpha_value,
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    tCcC_base=tCcC,
                    mC_mnl=mC_mnl,
                    overlapping_accum=self.overlapping_accum,
                    has_next_tile=work_tile.is_valid_tile,
                    pace_ns=cutlass.Int32(k_tile_cnt) * self.epi_store_pace_per_ktile_ns
                    if cutlass.const_expr(getattr(self, "epi_store_pace_per_ktile_ns", 0))
                    else None,
                    evict_first_ok=cute.size(mA_mkl, mode=[1])
                    >= self.epi_store_evict_first_min_k_bytes,
                )
            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)
            cute.arch.mbarrier_init_fence()
        griddepcontrol_launch_dependents()

    def _ab_descriptor(self, s, current, next_idx):
        """Pack stage addresses into an otherwise constant SW128 descriptor."""
        cur = s[None, 0, 0, 0]
        base = tcgen05.smem_descriptor_to_int(
            tcgen05.make_umma_smem_desc(cur.iterator, cur.layout, "k", next_src=cur.iterator)
        )
        stage_units = s.stride[3] // 16
        return (
            base
            + cutlass.Int64(current) * stage_units
            + cutlass.Int64(next_idx) * (stage_units << 16)
        )

    @staticmethod
    def _mma_prebuilt(tiled_mma, acc, ad, bd, offset):
        a_desc = tcgen05.int_to_smem_descriptor(ad + cutlass.Int64(offset))
        b_desc = tcgen05.int_to_smem_descriptor(bd + cutlass.Int64(offset))
        layout = cute.make_layout(1, stride=0)
        cute.mma_atom_call(
            tiled_mma, acc, cute.make_tensor(a_desc, layout), cute.make_tensor(b_desc, layout), acc
        )

    @staticmethod
    def make_desc_and_call_mma(
        tiled_mma: cute.TiledMma,
        d: cute.Tensor,
        sA_cur: cute.Tensor,
        sA_next: cute.Tensor,
        sB_cur: cute.Tensor,
        sB_next: cute.Tensor,
        c: cute.Tensor,
    ) -> None:
        """Specialized GEMM for circular-buffered A/B from SMEM.

        Performs D <- A * B + C where A and B are described by circular SMEM
        descriptors constructed from the (current, next) buffers. C and D may alias.

        Some tcgen05 MMAs require explicitly toggling an accumulate field outside of
        this routine; the caller is responsible for that.

        All tensors must already be partitioned for the provided tiled MMA.

        For MMA Atoms that require single-threaded execution, the gemm op automatically handles thread
        election internally. Manual thread selection is not required in such cases.

        :param atom: MMA atom
        :type atom: cute.MmaAtom
        :param d: Destination tensor
        :type d: cute.Tensor
        :param sA_cur: Current shared memory tensor for operand A
        :type sA_cur: cute.Tensor
        :param sA_next: Next shared memory tensor for operand A, used for circular buffering
        :type sA_next: cute.Tensor
        :param sB_cur: Current shared memory tensor for operand B
        :type sB_cur: cute.Tensor
        :param sB_next: Next shared memory tensor for operand B, used for circular buffering
        :type sB_next: cute.Tensor
        :param c: Third source tensor
        :type c: cute.Tensor
        :return: None
        :rtype: None
        """
        a_desc = tcgen05.make_umma_smem_desc(
            sA_cur.iterator,
            sA_cur.layout,
            "k" if tiled_mma.op.a_major_mode.name == "K" else "mn",
            next_src=sA_next.iterator,
        )
        b_desc = tcgen05.make_umma_smem_desc(
            sB_cur.iterator,
            sB_cur.layout,
            "k" if tiled_mma.op.b_major_mode.name == "K" else "mn",
            next_src=sB_next.iterator,
        )
        view_layout = cute.make_layout(1, stride=0)
        a_tensor = cute.make_tensor(a_desc, view_layout)
        b_tensor = cute.make_tensor(b_desc, view_layout)
        return cute.mma_atom_call(tiled_mma, d, a_tensor, b_tensor, c)

    @staticmethod
    def sm103_make_blockscaled_trivial_tiled_mma(
        sf_dtype: Type[cutlass.Numeric],
        cta_group: tcgen05.CtaGroup,
        mma_tiler_mn: Tuple[int, int],
        a_source: tcgen05.OperandSource = tcgen05.OperandSource.SMEM,
    ) -> cute.TiledMma:
        """Construct the SM103 K=96 NVFP4 MMA."""
        mma_op = tcgen05.SM103MmaMXF4NVF4Op(sf_dtype, (*mma_tiler_mn, 96), cta_group, a_source)
        return cute.make_tiled_mma(cute.make_mma_atom(mma_op))

    @staticmethod
    def sm103_make_smem_layout_a(
        tiled_mma: cute.TiledMma, mma_tiler_mnk: cute.Tile, num_stages: int
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        """
        Create the SMEM layout for operand A using K_SW128 and Uint8.

        This function creates a SMEM layout for operand A using the make_smem_layout_atom function with K_SW128 kind and Uint8 element type.

        :param tiled_mma: The tiled MMA atom
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The mma tiler shape (M, N, K)
        :type mma_tiler_mnk: cute.Tile
        :param num_stages: The number of stages
        :type num_stages: int

        :return: SMEM layout for operand A
        :rtype: cute.Layout
        """
        is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Uint8),
            cute.append(
                ((mma_tiler_mnk[0] // cute.size(tiled_mma.thr_layout_vmnk.shape[0]), 16), 1, 8),
                num_stages,
            ),
            order=(1, 0, 2) if not is_k_major else (0, 1, 2),
        )
        return a_smem_layout_staged

    @staticmethod
    def sm103_make_smem_layout_b(
        tiled_mma: cute.TiledMma, mma_tiler_mnk: cute.Tile, num_stages: int
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        """
        Create the SMEM layout for operand B using K_SW128 and Uint8.

        This function creates a SMEM layout for operand B using the make_smem_layout_atom function with K_SW128 kind and Uint8 element type.

        :param tiled_mma: The tiled MMA atom
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The mma tiler shape (M, N, K)
        :type mma_tiler_mnk: cute.Tile
        :param num_stages: The number of stages
        :type num_stages: int

        :return: SMEM layout for operand B
        :rtype: cute.Layout
        """
        is_k_major = tiled_mma.op.b_major_mode == OperandMajorMode.K
        b_smem_layout_staged = tcgen05.tile_to_mma_shape(
            tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Uint8),
            cute.append(
                ((mma_tiler_mnk[1] // cute.size(tiled_mma.thr_id.shape), 16), 1, 8), num_stages
            ),
            order=(1, 0, 2) if not is_k_major else (0, 1, 2),
        )
        return b_smem_layout_staged

    @dataclass(frozen=True)
    class Sm103BlockScaledBasicChunk:
        """
        Basic scale-factor atom layout decided by tcgen05 BlockScaled MMA Ops on SM103.

        Represents the fixed layout pattern for scale factors used by tcgen05
        BlockScaled MMA Ops on SM103. The layout is determined by the instruction
        specification and is not configurable.
        """

        sf_vec_size: int
        major_mode: OperandMajorMode = OperandMajorMode.K
        _layout: cute.Layout = field(init=False, repr=False)

        def __post_init__(self) -> None:
            if self.major_mode == OperandMajorMode.K:
                atom_shape = ((8, 4, 4), (16, 4))
                atom_stride = ((16, 128, 4), (0, 1))
            else:
                atom_shape = ((16, 4), (8, 4, 4))
                atom_stride = ((0, 1), (16, 128, 4))
            object.__setattr__(
                self, "_layout", cute.make_layout(shape=atom_shape, stride=atom_stride)
            )

        @property
        def layout(self) -> cute.Layout:
            return self._layout

    @staticmethod
    def sm103_make_smem_layout_sfa(
        tiled_mma: cute.TiledMma, mma_tiler: cute.Tile, num_stages: int
    ) -> cute.Layout:
        """Build the NVFP4 scale ring: four K=192 chunks per outer iteration."""
        mma_shape_mk = tiled_mma.partition_shape_A((mma_tiler[0], mma_tiler[2]))
        sf_atom = Sm103BlockScaledPersistentDenseGemmKernel.Sm103BlockScaledBasicChunk(
            16, tiled_mma.op.a_major_mode
        ).layout
        k_divisor = 4 or 4
        mma_sfa_tiler = (
            mma_shape_mk[0][0] * mma_shape_mk[1],
            mma_shape_mk[0][1] * mma_shape_mk[2] // k_divisor,
        )
        sfa_smem_atom_layout = cute.tiled_product(
            sf_atom,
            cute.make_layout(cute.shape_div(mma_sfa_tiler, cute.product_each(sf_atom.shape))),
        )
        sfa_smem_layout_staged = cute.make_layout(
            shape=cute.append(sfa_smem_atom_layout.shape, num_stages),
            stride=cute.append(
                sfa_smem_atom_layout.stride, cute.size(cute.filter_zeros(sfa_smem_atom_layout))
            ),
        )
        return sfa_smem_layout_staged

    @staticmethod
    def sm103_make_smem_layout_sfb(
        tiled_mma: cute.TiledMma, mma_tiler: cute.Tile, num_stages: int
    ) -> cute.Layout:
        """Build the NVFP4 scale ring: four K=192 chunks per outer iteration."""
        sf_atom = Sm103BlockScaledPersistentDenseGemmKernel.Sm103BlockScaledBasicChunk(
            16, tiled_mma.op.a_major_mode
        ).layout
        k_divisor = 4 or 4
        mma_sfb_tiler = (mma_tiler[1], mma_tiler[2] // k_divisor)
        if mma_sfb_tiler[0] == 128:
            sfb_smem_atom_layout = cute.tiled_product(
                sf_atom,
                cute.make_layout(cute.shape_div(mma_sfb_tiler, cute.product_each(sf_atom.shape))),
            )
        else:
            sf_k_major_atom256 = cute.make_layout(
                shape=((32, 4, 2), (16, 4)),
                stride=((16, 4, mma_sfb_tiler[1] // 16 // 4 * 512), (0, 1)),
            )
            sfb_smem_atom_layout = cute.tiled_product(
                sf_k_major_atom256,
                cute.make_layout(
                    cute.shape_div(mma_sfb_tiler, cute.product_each(sf_k_major_atom256.shape))
                ),
            )
        sfb_smem_layout_staged = cute.make_layout(
            shape=cute.append(sfb_smem_atom_layout.shape, num_stages),
            stride=cute.append(
                sfb_smem_atom_layout.stride, cute.size(cute.filter_zeros(sfb_smem_atom_layout))
            ),
        )
        return sfb_smem_layout_staged

    def mainloop_s2t_copy_and_partition(
        self, sSF: cute.Tensor, tSF: cute.Tensor
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        tCtSF_compact_copy = cute.make_tensor(
            tCtSF_compact.iterator,
            cute.append(
                cute.append(tCtSF_compact[None, 0, 0].layout, cute.make_layout(1)),
                cute.make_layout(1),
            ),
        )
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact_copy)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        sf_dtype: Type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Fit independent AB and SF rings in shared memory; output uses direct stores."""
        num_acc_stage = 1 if mma_tiler[1] == 256 else 2
        a_one = Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_a(
            tiled_mma, mma_tiler, 1
        )
        b_one = Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_b(
            tiled_mma, mma_tiler, 1
        )
        sfa_one = Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_sfa(
            tiled_mma, mma_tiler, 1
        )
        sfb_one = Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_sfb(
            tiled_mma, mma_tiler, 1
        )
        ab_bytes = cute.size_in_bytes(cutlass.Uint8, a_one) + cute.size_in_bytes(
            cutlass.Uint8, b_one
        )
        sf_bytes = cute.size_in_bytes(sf_dtype, sfa_one) + cute.size_in_bytes(sf_dtype, sfb_one)
        num_ab_stage = (smem_capacity // occupancy - (1024 + sf_bytes)) // ab_bytes
        num_sf_stage = (smem_capacity - occupancy * ab_bytes * num_ab_stage - occupancy * 1024) // (
            occupancy * sf_bytes
        )
        return (num_acc_stage, num_ab_stage, num_sf_stage)

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        swizzle_size: int = 1,
        raster_along_m: bool = True,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[0, (None, None, None)].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, swizzle_size, raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return (tile_sched_params, grid)

    @staticmethod
    def append_coalesce_layout(layout):
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        result = cute.append(part1, part2)
        result = cute.append(result, layout[3])
        result = cute.append(result, layout[4])
        result = cute.append(result, layout[5])
        return result

    @staticmethod
    def adapt_layout_for_tma_ab(composed_layout):
        layout = composed_layout.outer
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        part3 = cute.append(part2, layout[3])
        result = cute.append(part1, part3)
        return cute.make_composed_layout(composed_layout.inner, composed_layout.offset, result)

    @staticmethod
    def adapt_layout_for_tma_sf(layout):
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        result = cute.append(cute.group_modes(part1, 0, cute.rank(part1)), part2)
        return result

    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the wrapped GEMM kernel with dynamically shaped tensors.

        Uses TVM-FFI for efficient tensor passing: A, B, C, and alpha are passed
        as cute.Tensor directly (torch tensors at runtime via TVM-FFI's C-level
        dlpack). Scale factor tensors remain as pointers (complex 6D layout).

        Args:
            mA (cute.Tensor): Input A, shape (m, k_packed), Uint8 (FP4 packed).
            mB (cute.Tensor): Input B, shape (n, k_packed), Uint8 (FP4 packed).
            mC (cute.Tensor): Output C, shape (m, n).
            sf_m/sf_n/sf_k: Scale factor dimensions.
            l: Batch dimension.
            a_sf_ptr/b_sf_ptr: Scale factor pointers (6D layout).
            alpha_tensor: Alpha scaling factor, shape (1,), float32.
            max_active_clusters: Max active clusters.
            current_stream: CUDA stream (TVM-FFI fake stream).
            epilogue_op: Elementwise epilogue function.
        """
        m = cute.size(mA, mode=[0])
        k_packed = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])
        k = k_packed * 2
        a_fp4_ptr = cute.recast_ptr(mA.iterator, dtype=cutlass.Float4E2M1FN)
        a_tensor = cute.make_tensor(
            a_fp4_ptr, layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2))
        )
        b_fp4_ptr = cute.recast_ptr(mB.iterator, dtype=cutlass.Float4E2M1FN)
        b_tensor = cute.make_tensor(
            b_fp4_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        c_n = cute.assume(n, divby=64)
        c_tensor = cute.make_tensor(
            mC.iterator, layout=cute.make_ordered_layout((m, c_n, l), order=(1, 0, 2))
        )
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout((32, 4, sf_m, 4, sf_k, l), order=(2, 1, 4, 0, 3, 5)),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout((32, 4, sf_n, 4, sf_k, l), order=(2, 1, 4, 0, 3, 5)),
        )
        self(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            max_active_clusters,
            current_stream,
            epilogue_op,
        )
