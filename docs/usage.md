# Usage guide

[Installation](../README.md#setup) · [Complete example](../README.md#getting-started)

## 1. Prepare the inputs

The operation is `C = alpha * dequant(A) @ dequant(B).T`. A is `[M, K]`; B is **`[N, K]`**. N and K must be multiples of 64; M can be any size.

```python
a_q, a_sf, _ = quantize_nvfp4(a)
b_q, b_sf, _ = quantize_nvfp4(b)
```

Quantize once and reuse the packed tensors while their source values remain unchanged. The third return value contains linear scale factors for reference dequantization; GEMM uses the second, swizzled value. `quantize_nvfp4` uses a global scale of 1 by default; if you pass `global_scale`, set `alpha` to `1 / (global_scale_a * global_scale_b)`.

If your inputs are already NVFP4 in this layout, pass them directly:

| Argument | Shape / layout | Dtype |
|---|---|---|
| `a_q` | Contiguous `[M, K/2]`; two E2M1 values per byte, low nibble first | `torch.uint8` |
| `b_q` | Contiguous `[N, K/2]`, same packing | `torch.uint8` |
| `a_sf`, `b_sf` | Contiguous, flat, 16-byte aligned; one E4M3 scale per 16 K values in the quantizer's 128×4 swizzled layout, `ceil(rows/128)*128 * ceil(K/64)*4` elements (rows is M or N) | `torch.uint8` or `torch.float8_e4m3fn` |

## 2. Select the preset and run

```python
cfg = config_for(m, n, k)
with persisting_l2(cfg.persist_mb):
    c = gemm(a_q, b_q, a_sf, b_sf, config=cfg)
```

`config_for` selects the tuned preset for the four square sizes, or the default configuration (`Config.make()`) for other shapes. Keep the L2 context around a group of calls. It sets the host persisting-L2 limit and restores the previous limit when the context exits.

| Square size | `cfg.persist_mb` (MiB) |
|---|---:|
| 4096 | 0 |
| 8192 | 48 |
| 16384 | 40 |
| 32768 | 48 |

The first call compiles the kernel; later calls with the same configuration, output dtype, and device reuse it. The 8K, 16K, and 32K presets check for their exact shape and cluster layout and raise `ValueError` otherwise.

## 3. Choose the output

The default output is FP16. For BF16:

```python
with persisting_l2(cfg.persist_mb):
    c = gemm(a_q, b_q, a_sf, b_sf, config=cfg, out_dtype=torch.bfloat16)
```

For scaling and output-buffer reuse:

```python
alpha = torch.tensor([0.5], device=a_q.device, dtype=torch.float32)
c = torch.empty((m, n), device=a_q.device, dtype=torch.float16)

with persisting_l2(cfg.persist_mb):
    gemm(a_q, b_q, a_sf, b_sf, alpha=alpha, out=c, config=cfg)
```

`out` must be a contiguous `[M, N]` tensor with 32-byte-aligned storage. Its dtype must match `out_dtype`, which is not inferred from `out`; pass `out_dtype=torch.bfloat16` as well when reusing a BF16 buffer. A mismatched dtype, stride, or alignment raises `ValueError`.

`alpha` must be a one-element CUDA tensor (it is converted to float32); Python floats are not accepted. It scales the product after dequantization and defaults to 1.

## 4. Custom configurations

For tuning, create a `Config` with `Config.make()` and pass it to `gemm`. Start with the [existing presets](../nvfp4_gemm_cute/presets.py) and the [kernel design guide](kernel.md). Use the [benchmark harness](benchmarks.md) to compare configurations after compilation, with the required L2 reservation active.

[Kernel design](kernel.md) · [Benchmarks](benchmarks.md) · [Back to README](../README.md)
