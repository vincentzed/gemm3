"""NVFP4 quantization in PyTorch, in the layout the GEMM consumes.

NVFP4 stores E2M1 values with one E4M3 scale per 16 consecutive elements along K. Two values are
packed per byte, element 2j in the low nibble. Scale factors are handed to the kernel in the
128x4 swizzled layout: rows padded to 128 and scale columns to 4, then each (128 rows, 4 scale
columns) tile is stored as [32][4][4] (row % 32, row // 32, column).
"""

import torch

SF_VEC = 16
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _round_e2m1(x):
    """Round to the nearest E2M1 value (exact ties go to the even code) and return 4-bit codes.

    Matches flashinfer's nvfp4_quantize except on exact ties, where its reciprocal-multiply
    lands just above the midpoint.
    """
    mag = x.abs().clamp(max=6.0)
    table = _E2M1.to(x.device)
    dist = (mag.unsqueeze(-1) - table).abs()
    # Break ties toward even codes (odd codes get a tiny penalty).
    dist = dist + (torch.arange(8, device=x.device) % 2) * 1e-6
    code = dist.argmin(-1).to(torch.uint8)
    sign = x < 0  # negative values that round to zero keep the sign (-0), like cvt.rn.e2m1x2
    return code | (sign.to(torch.uint8) << 3)


def swizzle_sf(sf):
    """Convert [rows, cols] E4M3 scale factors (as uint8) to the 128x4 swizzled layout (flat)."""
    rows, cols = sf.shape
    rp, cp = -(-rows // 128) * 128, -(-cols // 4) * 4
    pad = torch.zeros(rp, cp, dtype=torch.uint8, device=sf.device)
    pad[:rows, :cols] = sf
    t = pad.view(rp // 128, 4, 32, cp // 4, 4).permute(0, 3, 2, 1, 4)
    return t.contiguous().view(-1)


def quantize_nvfp4(x, global_scale: float = 1.0):
    """Quantize a [rows, K] float tensor to NVFP4.

    Returns:
        (packed, sf_swizzled, sf_linear): packed is [rows, K/2] uint8; sf_swizzled is the flat
        128x4 layout the kernel reads; sf_linear is [rows, K/16] uint8 (E4M3 bits) for reference
        dequantization.
    """
    rows, k = x.shape
    assert k % SF_VEC == 0, "K must be a multiple of 16"
    xb = x.float().view(rows, k // SF_VEC, SF_VEC) * global_scale
    amax = xb.abs().amax(-1, keepdim=True)
    scale = (amax / 6.0).to(torch.float8_e4m3fn)
    denom = scale.float()
    q = torch.where(denom > 0, xb / denom, torch.zeros_like(xb))
    codes = _round_e2m1(q).view(rows, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    sf_linear = scale.view(rows, k // SF_VEC).view(torch.uint8)
    return packed.contiguous(), swizzle_sf(sf_linear), sf_linear.contiguous()


def dequantize_nvfp4(packed, sf_linear, global_scale: float = 1.0):
    """Expand packed NVFP4 plus linear scale factors back to a float32 [rows, K] tensor."""
    rows = packed.shape[0]
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], -1).view(rows, -1)
    table = _E2M1.to(packed.device)
    val = table[(codes & 7).long()] * torch.where((codes & 8) > 0, -1.0, 1.0)
    scale = sf_linear.view(torch.float8_e4m3fn).float().repeat_interleave(SF_VEC, dim=1)
    return val * scale / global_scale
