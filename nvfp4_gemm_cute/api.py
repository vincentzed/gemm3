"""Compile and launch the SM103 NVFP4 GEMM.

``gemm(a, b, a_sf, b_sf)`` computes ``C = alpha * dequant(A) @ dequant(B).T`` with A [M, K/2] and
B [N, K/2] packed NVFP4 (uint8, two values per byte) and scale factors in the 128x4 swizzled
layout (see ``quant.swizzle_sf``). Kernels are cached per configuration. General configurations
accept dynamic shapes; specialized presets validate their exact shapes before compilation.
"""

import contextlib
import warnings

import torch

from .hw import arch, get_max_active_clusters
from .presets import config_for

_CACHE = {}
_ALPHA_ONE = {}

_SUPPORTED_ATTRIBUTES = frozenset(
    {
        "fast_swz",
        "raster_along_m",
        "owned_8k",
        "l2_policy_a",
        "l2_policy_b",
        "l2_policy_sfa",
        "l2_policy_sfb",
        "epi_store_pace_per_ktile_ns",
        "prefetch_first_ktiles",
        "trim_short_tail",
        "tma_b_first",
        "tma_sfb_first",
        "retire_a_cols",
        "packed_ab_desc",
        "epi_store_pair",
        "specialize_32k",
    }
)


def supported() -> bool:
    """Return True if the current GPU is sm_103 (B300/GB300) and the DSL has the needed ops."""
    if not torch.cuda.is_available() or arch() != "sm_103":
        return False
    try:
        from cutlass.cute.nvgpu.tcgen05 import SM103MmaMXF4NVF4Op  # noqa: F401
    except ImportError:
        return False
    return True


def _cutlass_dtype(dtype):
    import cutlass

    table = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}
    if dtype not in table:
        raise ValueError(f"unsupported output dtype {dtype}; use float16 or bfloat16")
    return table[dtype]


def _num_m_clusters(m, cfg):
    cta_m = 128  # 2-CTA 256-row tiles and 1-CTA 128-row tiles both use 128-row CTA tiles
    return -(-(-(-m // cta_m)) // cfg.cluster[0])


def _effective_attrs(m, cfg):
    """Return the kernel attributes for this launch, with fast_swz dropped if M does not allow it.

    fast_swz decodes groups of fast_swz M-clusters; when the M-cluster count is not a multiple of
    it, tiles in the last partial group would map outside the grid and be skipped.
    """
    attrs = cfg.attr_dict()
    unknown = set(attrs) - _SUPPORTED_ATTRIBUTES
    if unknown:
        raise ValueError(f"unsupported kernel attributes: {', '.join(sorted(unknown))}")
    s = attrs.get("fast_swz", 1)
    if s > 1:
        if s & (s - 1):
            raise ValueError(f"fast_swz must be a power of two, got {s}")
        if _num_m_clusters(m, cfg) % s:
            warnings.warn(
                f"fast_swz={s} needs the M-cluster count ({_num_m_clusters(m, cfg)}) to be a "
                "multiple of it; falling back to plain raster order",
                stacklevel=3,
            )
            attrs["fast_swz"] = 1
        attrs["raster_along_m"] = False
    return attrs


def _compile(cfg, attrs, out_dtype, enable_pdl, device_index):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from .kernel import Sm103BlockScaledPersistentDenseGemmKernel as K103

    key = (
        cfg.tile,
        cfg.cluster,
        cfg.fallback,
        tuple(sorted(attrs.items())),
        out_dtype,
        enable_pdl,
        device_index,
    )
    if key in _CACHE:
        return _CACHE[key]
    gemm = K103(cfg.tile, cfg.cluster, enable_pdl, fallback_cluster_shape_mn=cfg.fallback)
    for name, value in attrs.items():
        setattr(gemm, name, value)
    if attrs.get("owned_8k") and gemm.mixed_num_slots != 9:
        raise ValueError("owned_8k scheduling requires nine persistent cluster slots")
    max_active = get_max_active_clusters(cfg.cluster[0] * cfg.cluster[1])
    sm, sk, sn = cute.sym_int(), cute.sym_int(), cute.sym_int()
    if attrs.get("specialize_32k"):
        sm, sk, sn = 32768, 16384, 32768
    a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sm, sk), stride_order=(1, 0), assumed_align=32
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sn, sk), stride_order=(1, 0), assumed_align=32
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        _cutlass_dtype(out_dtype), (sm, sn), stride_order=(1, 0), assumed_align=32
    )
    sfa = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
    sfb = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        gemm.wrapper,
        a_fake,
        b_fake,
        c_fake,
        1,
        1,
        1,
        1,
        sfa,
        sfb,
        alpha_fake,
        max_active,
        stream,
        options=f"--opt-level {3 if attrs.get('specialize_32k') else 2} --enable-tvm-ffi",
    )
    _CACHE[key] = compiled
    return compiled


def gemm(
    a, b, a_sf, b_sf, alpha=None, out=None, out_dtype=torch.float16, config=None, enable_pdl=True
):
    """Compute ``alpha * dequant(a) @ dequant(b).T`` on sm_103.

    Args:
        a: [M, K/2] uint8 packed NVFP4 (row-major, K contiguous).
        b: [N, K/2] uint8 packed NVFP4 (the weight layout, K contiguous).
        a_sf, b_sf: E4M3 scale factors in the 128x4 swizzled layout (uint8 or float8_e4m3fn).
        alpha: optional float32 tensor with one element; defaults to 1.0.
        out: optional [M, N] output of out_dtype.
        config: a presets.Config; defaults to presets.config_for(M, N, K).

    Requires N % 64 == 0 and K % 64 == 0. A configuration with persist_mb > 0 only reaches its
    measured speed inside ``persisting_l2(config.persist_mb)``.
    """
    m, kp = a.shape
    n = b.shape[0]
    k = kp * 2
    if b.shape[1] != kp:
        raise ValueError(f"K mismatch: a is [{m}, {kp}], b is [{n}, {b.shape[1]}]")
    if n % 64 or k % 64:
        raise ValueError(f"N and K must be multiples of 64 (got N={n}, K={k})")
    cfg = config or config_for(m, n, k)
    attrs = _effective_attrs(m, cfg)
    if attrs.get("specialize_32k"):
        if (m, n, k) != (32768, 32768, 32768):
            raise ValueError("32K specialization requires M=N=K=32768")
        if cfg.tile != (256, 256) or cfg.cluster != (4, 4) or cfg.fallback != (2, 1):
            raise ValueError(
                "32K specialization requires tile=(256, 256), cluster=(4, 4), fallback=(2, 1)"
            )
    if attrs.get("owned_8k") and (m, n, k) != (8192, 8192, 8192):
        raise ValueError("owned_8k scheduling requires M=N=K=8192")
    if attrs.get("owned_8k") and (
        cfg.tile != (256, 256) or cfg.cluster != (8, 2) or cfg.fallback != (2, 1)
    ):
        raise ValueError(
            "owned_8k scheduling requires tile=(256, 256), cluster=(8, 2), fallback=(2, 1)"
        )
    if attrs.get("trim_short_tail"):
        if (m, n, k) != (16384, 16384, 16384):
            raise ValueError("16K optimizations require M=N=K=16384")
        if cfg.tile != (256, 256) or cfg.cluster != (8, 2) or cfg.fallback != (2, 1):
            raise ValueError(
                "16K optimizations require tile=(256, 256), cluster=(8, 2), fallback=(2, 1)"
            )
    if attrs.get("packed_ab_desc"):
        expected_cluster = {(16384, 16384, 16384): (8, 2), (32768, 32768, 32768): (4, 4)}
        cluster = expected_cluster.get((m, n, k))
        if cluster is None:
            raise ValueError("packed_ab_desc requires M=N=K=16384 or M=N=K=32768")
        if cfg.tile != (256, 256) or cfg.cluster != cluster or cfg.fallback != (2, 1):
            raise ValueError(
                f"packed_ab_desc requires tile=(256, 256), cluster={cluster}, fallback=(2, 1)"
            )
    dev = a.device
    compiled = _compile(cfg, attrs, out_dtype, enable_pdl, dev.index or 0)
    if out is None:
        out = torch.empty(m, n, dtype=out_dtype, device=dev)
    if alpha is None:
        alpha = _ALPHA_ONE.get(dev)
        if alpha is None:
            alpha = _ALPHA_ONE[dev] = torch.ones(1, dtype=torch.float32, device=dev)
    else:
        alpha = alpha.reshape(1).to(torch.float32)
    sf_m, sf_n, sf_k = -(-m // 128), -(-n // 128), -(-(k // 16) // 4)
    compiled(a, b, out, sf_m, sf_n, sf_k, a_sf.data_ptr(), b_sf.data_ptr(), alpha)
    return out


@contextlib.contextmanager
def persisting_l2(mb: int):
    """Set the persisting-L2 limit to ``mb`` MiB for this scope, then restore the old limit.

    This host limit (cudaLimitPersistingL2CacheSize) is separate from the kernel's operand
    eviction hints. It does not select addresses or guarantee residency. Restoration
    happens on context exit and does not synchronize with GPU completion.
    """
    from cuda.bindings import runtime as rt

    lim = rt.cudaLimit.cudaLimitPersistingL2CacheSize
    torch.cuda.init()
    err, old = rt.cudaDeviceGetLimit(lim)
    (err,) = rt.cudaDeviceSetLimit(lim, int(mb) << 20)
    if err != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaDeviceSetLimit(PersistingL2CacheSize, {mb} MB) failed: {err}")
    try:
        yield
    finally:
        rt.cudaDeviceSetLimit(lim, old)
