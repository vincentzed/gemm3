"""NVFP4 x NVFP4 GEMM for B300 (sm_103) in CuTe DSL.

gemm: C = alpha * dequant(A) @ dequant(B).T with per-shape tuned launch configurations.
quantize_nvfp4 / dequantize_nvfp4 / swizzle_sf: NVFP4 packing in the layout the kernel reads.
persisting_l2: scoped persisting-L2 reservation required by presets with persist_mb > 0.
"""

__version__ = "0.1.0"
from .api import gemm, persisting_l2, supported
from .presets import DEFAULT, PRESETS, Config, config_for
from .quant import dequantize_nvfp4, quantize_nvfp4, swizzle_sf

__all__ = [
    "DEFAULT",
    "PRESETS",
    "Config",
    "config_for",
    "dequantize_nvfp4",
    "gemm",
    "persisting_l2",
    "quantize_nvfp4",
    "supported",
    "swizzle_sf",
]
