"""Inputs and reference checks for the NVFP4 GEMM."""

import torch

from .quant import dequantize_nvfp4, quantize_nvfp4


def make_inputs(m, n, k, seed=0, device="cuda"):
    """Return quantized NVFP4 operands for a random [m, k] x [n, k] problem."""
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(m, k, device=device, generator=g)
    b = torch.randn(n, k, device=device, generator=g)
    a_q, a_sf, a_sfl = quantize_nvfp4(a)
    b_q, b_sf, b_sfl = quantize_nvfp4(b)
    return dict(a=a_q, b=b_q, a_sf=a_sf, b_sf=b_sf, a_sf_linear=a_sfl, b_sf_linear=b_sfl)


def reference(inp):
    """FP32 reference of dequant(A) @ dequant(B).T (TF32 disabled)."""
    a = dequantize_nvfp4(inp["a"], inp["a_sf_linear"])
    b = dequantize_nvfp4(inp["b"], inp["b_sf_linear"])
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return a @ b.T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def check(out, ref, rel_tol=2e-3):
    """Return (ok, relative L2 error) of out against ref."""
    err = (out.float() - ref).norm() / ref.norm().clamp_min(1e-30)
    return bool(err <= rel_tol and torch.isfinite(out).all()), float(err)
