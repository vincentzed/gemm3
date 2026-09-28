# Copyright (c) 2025 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
#
# This file contains modified versions of epilogue utility functions from
# NVIDIA CUTLASS DSL, adapted to support optimized output scaling.
#
# The key optimization is applying the output_scale to the accumulator
# BEFORE converting to the output dtype, which:
# 1. Avoids an extra type conversion (BFloat16 -> Float32 promotion)
# 2. Preserves precision by scaling in Float32 before the final conversion

"""Direct output epilogue: drain overlapping TMEM, scale in FP32, and store FP16/BF16."""

from typing import Tuple
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import Boolean, Constexpr, Int32, const_expr
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu.common import CacheEvictionPriority
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
    epilogue_tmem_copy_and_partition,
)

__all__ = ["epilogue_with_alpha"]


@cute.jit
def _st_global_ef_v8(addr_i64, v):
    """st.global.L1::no_allocate.L2::evict_first.v8.b32 (32 B)."""
    _llvm.inline_asm(
        None,
        [cutlass.Int64(addr_i64).ir_value()] + [cutlass.Int32(x).ir_value() for x in v],
        "st.global.L1::no_allocate.L2::evict_first.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
        "l,r,r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def epilogue_with_alpha(
    gemm_kernel,
    epi_tidx: Int32,
    tCtAcc_base: cute.Tensor,
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    epilogue_op: Constexpr,
    alpha_value: cutlass.Float32,
    mma_tile_coord_mnl: Tuple[Int32, Int32, Int32],
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    tCcC_base: cute.Tensor = None,
    mC_mnl: cute.Tensor = None,
    overlapping_accum: Constexpr = False,
    has_next_tile=None,
    pace_ns=None,
    evict_first_ok=None,
) -> pipeline.PipelineState:
    """
    Per-tile epilogue (direct store) that applies alpha scaling in Float32 BEFORE
    converting to c_dtype, preventing overflow for narrow types like Float16.

    This is a drop-in replacement for cutlass.utils.gemm.sm100.epilogue with an
    additional alpha_value parameter. The alpha is applied to the Float32 accumulator
    before the conversion to c_dtype.
    """
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )
    gC_epi = cute.flat_divide(tCgC, epi_tile)
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gC_partitioned = thr_copy_t2r.partition_D(gC_epi)
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC_partitioned[None, None, None, 0, 0, 0, 0, 0].shape, gemm_kernel.c_dtype
    )
    mclD = cute.max_common_layout(
        tTR_rC.layout, tTR_gC_partitioned[None, None, None, 0, 0, 0, 0, 0].layout
    )
    num_bits_per_copy = min(
        tTR_gC_partitioned.iterator.alignment * 8, cute.size(mclD) * gemm_kernel.c_dtype.width, 256
    )
    simt_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        gemm_kernel.c_dtype,
        num_bits_per_copy=num_bits_per_copy,
        l1c_evict_priority=getattr(
            gemm_kernel, "c_l1c_evict_priority", CacheEvictionPriority.NO_ALLOCATE
        ),
        store_cache_mode=getattr(
            gemm_kernel, "c_store_cache_mode", cute.nvgpu.StoreCacheMode.WRITE_BACK
        ),
    )
    use_predication = tCcC_base is not None and mC_mnl is not None
    if const_expr(use_predication):
        tCcC = transform_partitioned_tensor_layout(tCcC_base)
        cC_epi = cute.flat_divide(tCcC, epi_tile)
        tTR_cC_partitioned = thr_copy_t2r.partition_D(cC_epi)
    tTR_gC = tTR_gC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]
    if const_expr(use_predication):
        tTR_cC = tTR_cC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]
        tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))
    if const_expr(overlapping_accum):
        acc_stage_index = acc_consumer_state.phase
        reverse_subtile = acc_stage_index == 0
    else:
        acc_stage_index = acc_consumer_state.index
    tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, acc_stage_index]
    acc_pipeline.consumer_wait(acc_consumer_state)
    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    _rf = overlapping_accum
    if const_expr(_rf):
        # Drain every column the next accumulator phase will overwrite before release.
        n_pre = gemm_kernel.iter_acc_early_release_in_epilogue + 1
        pre = [cute.make_rmem_tensor_like(tTR_rAcc) for _ in range(n_pre)]
        for i in cutlass.range_constexpr(n_pre):
            ri = i
            if reverse_subtile:
                ri = subtile_cnt - 1 - i
            cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, ri], pre[i])
        # The register copies must complete before the producer can reuse TMEM.
        cute.arch.fence_view_async_tmem_load()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        hold = cute.make_rmem_tensor((cute.size(tTR_rC) // 2,), cutlass.Int32)
        hold_addr = cutlass.Int64(0)
        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
            real_subtile_idx = subtile_idx
            if reverse_subtile:
                real_subtile_idx = subtile_cnt - 1 - subtile_idx
            tTR_gC_subtile = tTR_gC[None, None, None, real_subtile_idx]
            rAcc = tTR_rAcc
            if const_expr(subtile_idx < n_pre):
                rAcc = pre[subtile_idx]
            else:
                cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, real_subtile_idx], rAcc)
            acc_vec = rAcc.load()
            acc_vec = epilogue_op((acc_vec * alpha_value).to(gemm_kernel.c_dtype))
            tTR_rC.store(acc_vec)
            if const_expr(gemm_kernel.c_layout == cutlass.utils.LayoutEnum.ROW_MAJOR):
                crd0 = tTR_cC[None, None, None, real_subtile_idx][0, 0, 0]
                n_cols = mC_mnl.shape[1]
                rC32 = cute.recast_tensor(tTR_rC, cutlass.Int32)
                nw = cute.size(rC32)
                fast = (
                    (crd0[0] < mC_mnl.shape[0]) & (crd0[1] + nw * 2 <= n_cols) & (n_cols % 16 == 0)
                )
                if const_expr(evict_first_ok is not None):
                    fast = fast & evict_first_ok
                if fast:
                    gaddr = tTR_gC_subtile.iterator.toint()
                    if const_expr(getattr(gemm_kernel, "epi_store_pair", False)):
                        # Keep the first half until its adjacent row segment is ready.
                        if const_expr(subtile_idx % 2 == 0):
                            for w in cutlass.range_constexpr(nw):
                                hold[w] = rC32[w]
                            hold_addr = gaddr
                        else:
                            for c in cutlass.range_constexpr(nw // 8):
                                _st_global_ef_v8(
                                    hold_addr + c * 32, [hold[c * 8 + w] for w in range(8)]
                                )
                            for c in cutlass.range_constexpr(nw // 8):
                                _st_global_ef_v8(
                                    gaddr + c * 32, [rC32[c * 8 + w] for w in range(8)]
                                )
                    else:
                        for c in cutlass.range_constexpr(nw // 8):
                            _st_global_ef_v8(gaddr + c * 32, [rC32[c * 8 + w] for w in range(8)])
                else:
                    tTR_cC_subtile = tTR_cC[None, None, None, real_subtile_idx]
                    pred_C_shape = (1, *tTR_cC_subtile.shape[1:])
                    pred_C = cute.make_rmem_tensor(pred_C_shape, Boolean)
                    for m_idx in range(tTR_cC_subtile.shape[1]):
                        for n_idx in range(tTR_cC_subtile.shape[2]):
                            vector_first_coord = tTR_cC_subtile[0, m_idx, n_idx]
                            pred_C[0, m_idx, n_idx] = cute.elem_less(
                                vector_first_coord, mC_mnl.shape
                            )
                    cute.copy(simt_atom, tTR_rC, tTR_gC_subtile, pred=pred_C)
            elif const_expr(use_predication):
                tTR_cC_subtile = tTR_cC[None, None, None, real_subtile_idx]
                pred_C_shape = (1, *tTR_cC_subtile.shape[1:])
                pred_C = cute.make_rmem_tensor(pred_C_shape, Boolean)
                for m_idx in range(tTR_cC_subtile.shape[1]):
                    for n_idx in range(tTR_cC_subtile.shape[2]):
                        vector_first_coord = tTR_cC_subtile[0, m_idx, n_idx]
                        pred_C[0, m_idx, n_idx] = cute.elem_less(vector_first_coord, mC_mnl.shape)
                cute.copy(simt_atom, tTR_rC, tTR_gC_subtile, pred=pred_C)
            else:
                cute.copy(simt_atom, tTR_rC, tTR_gC_subtile)
            if const_expr(pace_ns is not None and subtile_idx < subtile_cnt - 1):
                t_end = cute.arch.globaltimer() + cutlass.Int64(pace_ns)
                # There is no successor mainloop to protect on the last tile.
                if not has_next_tile:
                    t_end = cute.arch.globaltimer()
                while cute.arch.globaltimer() < t_end:
                    _llvm.inline_asm(
                        None,
                        [],
                        "nanosleep.u32 100;",
                        "",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=_llvm.AsmDialect.AD_ATT,
                    )
    else:
        for subtile_idx in range(subtile_cnt):
            real_subtile_idx = subtile_idx
            if const_expr(overlapping_accum):
                if reverse_subtile:
                    real_subtile_idx = subtile_cnt - 1 - subtile_idx
            tTR_gC_subtile = tTR_gC[None, None, None, real_subtile_idx]
            tTR_tAcc_mn = tTR_tAcc[None, None, None, real_subtile_idx]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            if const_expr(overlapping_accum):
                if subtile_idx == gemm_kernel.iter_acc_early_release_in_epilogue:
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
            elif subtile_idx == subtile_cnt - 1:
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
            acc_vec = tTR_rAcc.load()
            acc_vec = epilogue_op((acc_vec * alpha_value).to(gemm_kernel.c_dtype))
            tTR_rC.store(acc_vec)
            if const_expr(use_predication):
                tTR_cC_subtile = tTR_cC[None, None, None, real_subtile_idx]
                pred_C_shape = (1, *tTR_cC_subtile.shape[1:])
                pred_C = cute.make_rmem_tensor(pred_C_shape, Boolean)
                for m_idx in range(tTR_cC_subtile.shape[1]):
                    for n_idx in range(tTR_cC_subtile.shape[2]):
                        vector_first_coord = tTR_cC_subtile[0, m_idx, n_idx]
                        pred_C[0, m_idx, n_idx] = cute.elem_less(vector_first_coord, mC_mnl.shape)
                cute.copy(simt_atom, tTR_rC, tTR_gC_subtile, pred=pred_C)
            else:
                cute.copy(simt_atom, tTR_rC, tTR_gC_subtile)
    return acc_consumer_state
