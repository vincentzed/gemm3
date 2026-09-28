# Benchmarks

[Results](#library-comparison) · [Methodology](#measurement-conditions) · [Implementations](#implementations) · [Versions](#versions-and-source) · [Reproduce](#reproduce) · [Other libraries](#other-libraries) · [Figures](#recreate-the-figures)

## Library comparison

This page compares eight implementations on the **same B300 (GPU UUID recorded in each result)**. See [implementations](#implementations) and [versions and source](#versions-and-source).

Median kernel latency in **µs** (lower is better). † BF16 output.

| Implementation | 4096³ | 8192³ | 16384³ | 32768³ |
|---|---:|---:|---:|---:|
| GEMM³ | 22.05 | 119.15 | 960.30 | 8210.53 |
| fast.cu | 24.92 | 131.42 | 1049.10 | 8633.83 |
| cuBLASLt | 25.07 | 131.94 | 1044.08 | 8775.88 |
| CUTLASS | 28.08 | 129.69 | 1177.26 | 9353.03 |
| cuDNN | 25.50 | 132.30 | 1043.99 | 8774.67 |
| FlashInfer CuTe DSL | 27.50 | 155.92 | 1612.14 | 11564.31 |
| QuACK | 28.86 | 154.99 | 1202.74 | 9939.03 |
| ThunderKittens † | 31.74 | 147.19 | 1128.26 | 9282.67 |

Raw data is in the JSONs: [4096³](../benchmarks/results/libraries-4096.json) · [8192³](../benchmarks/results/libraries-8192.json) · [16384³](../benchmarks/results/libraries-16384.json) · [32768³](../benchmarks/results/libraries-32768.json).

Limitation: these square-shape GEMMs are benchmark shapes, not shapes used in real serving.

## Measurement conditions

- All measurements require **CUPTI 13.4**. Missing CUPTI triggers a hard error, as falling back to CUDA events is not permitted.

- Measurements use CUDA graph replay with a cooldown period between samples, and the L2 cache is flushed outside of the timed window. The GEMM³ preset receives its configured persisting-L2 cache reservation, while all other implementations run without an external reservation.

- Each benchmarking round records the median of **50 timed replays**, and the final reported latency is the median of **eight round medians**. To prevent systematic bias, the execution order of the tested libraries is rotated and reversed between rounds.

- CUPTI measures the total GPU activity span of a single captured call, including any auxiliary kernels if an implementation launches more than one. Overhead tasks such as quantization, allocation, padding, repacking, compilation, and autotuning are excluded from the timed window. (While library autotuners may use their own timing methods, final reported measurements use CUPTI.)

- Before launch, the target GPU is verified to have zero memory usage, zero utilization, and no active compute processes. The test harness re-verifies that no unrelated processes have started before every individual timing round, though other GPUs on the host machine may remain occupied.

- No modifications are made to GPU clock speeds or power limits. Observed throughput can vary depending on the specific GPU, memory allocation layout, active clock states, and test harness.

### Throughput and precision

Standard dense throughput is calculated as `2 * M * N * K / (latency_us * 1e6)` in TFLOP/s (divide by 1000 for PFLOP/s). To ensure padding does not artificially inflate performance metrics, throughput is calculated using the **requested** dimensions. This matters for ThunderKittens, whose inputs the harness pads to a multiple of 768 in `K` before timing.

ThunderKittens outputs in BF16, while all other implementations output in FP16; this precision difference is marked on the plot. Finally, the error bars (whiskers) on the plot represent the observed throughput range, derived from the slowest and fastest median times across benchmarking rounds.

### Correctness

Before timing, every implementation's full output is compared with GEMM³'s output at the same precision. The raw JSON records bit equality and relative L2-norm error. GEMM³'s output is also checked independently against an FP32 matmul (dequantized inputs, TF32 disabled) on a 128×128 sample: the intersection of 128 sampled rows and 128 sampled columns.

## Implementations

The comparison uses the same packed NVFP4 operands and E4M3 scales for every implementation. All inputs have **16-element scale blocks** and `alpha=1`. This is dense W4A4 GEMM.

| Plot label | Implementation and selection | Output |
|---|---|---|
| GEMM³ | This package's shape preset and configured L2 reservation | FP16 |
| fast.cu | Upstream `gb300/nvfp4`, final kernel revision `r9`; screen all nine schedules with CUPTI, then measure the fastest | FP16 |
| cuBLASLt | First heuristic with a 64 MiB workspace limit, matching the fast.cu article's policy | FP16 |
| CUTLASS | FlashInfer's SM103 CUTLASS backend, autotuned before measurement | FP16 |
| cuDNN | FlashInfer's cuDNN backend, autotuned before measurement | FP16 |
| FlashInfer CuTe DSL | Upstream main `mm_fp4(backend="cute-dsl")` at the `sources.json` revision, which does not include PR #4866; autotuned before measurement | FP16 |
| QuACK | Upstream `gemm` with `BlockScaledOperand(format="nvfp4")`, `tuned=True` | FP16 |
| ThunderKittens | Unmodified `kernels/gemm/nvfp4_b300` kernel | BF16 |

**ThunderKittens uses NVFP4 inputs and BF16 output** (no FP16 output available). It is marked † on the plot. Other rows return FP16.

ThunderKittens consumes complete `K=768` tiles. Zero-padding changes K from 4096 / 8192 / 16384 / 32768 to **4608 / 8448 / 16896 / 33024**, which is 12.5%, 3.1%, 3.1%, and 0.8% more K work. Padding and repacking happen before timing; the additional GEMM work **is** timed. Reported throughput counts the original requested K. This is a limitation of the comparison.

The fast.cu source's route tables cover 4096 output tiles. The 32K case needs 16384. For **32K only**, the build helper enlarges the two route tables and places them in device global memory because they exceed constant-memory capacity. Smaller shapes use the unmodified upstream tables. The main GEMM computation is unchanged. This adaptation is explicit in [build_wrappers.py](../benchmarks/build_wrappers.py).

The [fast.cu article](https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4) uses a different measurement system. Its kernel and cuBLASLt are remeasured here with CUPTI, cold L2, and the repeated rounds described above.

## Versions and source

Measured with **CUDA 13.4.92, cuBLAS 13.8.0.4, cuDNN 9.26.0.51, CUTLASS DSL 4.8.0 and CUPTI Python 13.4.0**. cuDNN Frontend is built from the revision in `sources.json`, reporting 1.31.0. cuBLAS 13.8 comes from its own pip package, so its version is independent of the 13.4 toolkit. PyTorch 2.13.0+cu130 provides only tensor storage and graph capture; the GEMM libraries it runs are the ones above.

- [Source revisions](../benchmarks/results/sources.json)

- [Installed versions](../benchmarks/results/environment.json)

- Each result JSON records the actual loaded cuBLAS, cuDNN, NVRTC and CUPTI library paths.

Open-source GEMM implementations and C++ adapters are built from the recorded source revisions. CUTLASS and CCCL use FlashInfer's pinned submodules. CuTe DSL kernels are JIT-compiled from source. cuBLAS and cuDNN kernels are NVIDIA binaries only.

Install dependencies in isolated directories and select them through `PYTHONPATH`, `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH`, so shared installations stay unchanged.

## Reproduce

Use the commands below from the repository root. To compare the tuned presets with the default configuration (`Config.make()`):

```bash
pip install -e ".[cu13,bench,dev]"
CUDA_VISIBLE_DEVICES=<gpu> python benchmarks/bench.py --shapes 4096,8192,16384,32768
```

### Build and run

Use the commits in `sources.json` for reproducibility. Install CUDA-enabled PyTorch, CUTLASS DSL 4.8.0, CUPTI Python 13.4.0, and the NVIDIA libraries above. Point `PYTHONPATH`, `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH` at that toolkit and those libraries.

Clone the upstream projects listed in `sources.json`, checking out the SHAs noted. Initialize FlashInfer's `cutlass`, `cccl`, and `spdlog` submodules (required).

Building these libraries from source also needs their upstream build dependencies: a C++20 compiler, CMake, Ninja, Python development headers, TVM FFI, and pybind11. CUTLASS's reference headers require cuRAND headers. NVIDIA pip packages require unversioned `.so` symlinks when used as a standalone development toolkit.

From an isolated environment:

```bash
# SOURCE_DIR contains those checkouts; CUDNN_ROOT is the cuDNN installation above.
# BUILD_NVEP=0 skips FlashInfer's optional NIXL-EP and NCCL-EP backends.
BUILD_NVEP=0 pip install --no-deps --no-build-isolation "$SOURCE_DIR/flashinfer"
pip install --no-deps --no-build-isolation "$SOURCE_DIR/quack"
CUDAToolkit_ROOT="$CUDA_HOME" CUDNN_PATH="$CUDNN_ROOT" \
  pip install --no-deps --no-build-isolation "$SOURCE_DIR/cudnn-frontend"

python benchmarks/build_wrappers.py \
  --fastcu "$SOURCE_DIR/fast.cu" \
  --thunderkittens "$SOURCE_DIR/ThunderKittens" \
  --nvcc "$CUDA_HOME/bin/nvcc"
```

```bash
CUDA_VISIBLE_DEVICES=<gpu> python benchmarks/compare.py \
  --size 4096 --rounds 8 --output benchmarks/results/libraries-4096.json
```

Repeat for 8192, 16384, and 32768. All eight implementations run by default; their `--only` keys are `preset` (GEMM³), `fastcu`, `cublaslt`, `cutlass`, `cudnn`, `cute-dsl` (FlashInfer CuTe DSL), `quack`, and `thunderkittens`. Pass a subset, such as `--only preset,fastcu`, for a smaller comparison. Timing follows the [measurement conditions](#measurement-conditions).

## Other libraries

- DeepGEMM's native FP4 kernels use the MXFP4 scale format; its NVFP4 entry point wraps cuBLASLt, so it is excluded as a duplicate.

- TileLang's NVFP4 GEMM example targeted SM120 when these results were recorded, so it is excluded.

- Weight-only W4A16 kernels and grouped MoE GEMMs are different workloads and out of scope.

## Recreate the figures

CPU only; no GPU or PyTorch is needed:

```bash
uv run --no-project --with matplotlib==3.10.8 python figures/plot.py
python3 figures/kernel_diagrams.py  # standard library only
```

[Usage](usage.md) · [Kernel design](kernel.md) · [Back to README](../README.md)
