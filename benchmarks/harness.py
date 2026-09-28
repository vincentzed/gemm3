"""CUPTI timing under CUDA graph replay with a cold L2 (flashinfer's bench_gpu_time)."""

import statistics
import warnings
from importlib.metadata import version

from flashinfer.testing.utils import bench_gpu_time


def time_us(fn, iters=50, rounds=5, cold=True):
    """Median over rounds of the per-round median kernel time in microseconds.

    Each iteration flushes L2 (outside the timed window) and sleeps afterwards so the GPU does not
    run into its power cap between samples.
    """
    # Fail closed: FlashInfer otherwise silently substitutes CUDA-event timing.
    from cupti import cupti  # noqa: F401

    if int(version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("CUPTI >= 13 is required; event timing is not accepted")
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")
    meds = []
    for _ in range(rounds):
        t = bench_gpu_time(
            fn,
            dry_run_iters=max(3, iters // 4),
            repeat_iters=iters,
            enable_cupti=True,
            use_cuda_graph=True,
            cold_l2_cache=cold,
            sleep_after_run=True,
        )
        meds.append(statistics.median(t) * 1e3)
    return statistics.median(meds), min(meds), max(meds)
