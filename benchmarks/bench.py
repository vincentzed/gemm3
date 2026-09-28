"""Benchmark the tuned presets against the default configuration and, optionally, fast.cu.

    python benchmarks/bench.py --shapes 4096,8192,16384
    python benchmarks/bench.py --shapes 4096 --fastcu-lib /path/to/libfastcu.so

fast.cu (github.com/pranjalssh/fast.cu, gb300/nvfp4) is timed at the best of its tile schedules
and its output is compared bit-for-bit with ours.
"""

import argparse
import ctypes
import os
import statistics

import torch
from harness import time_us

from nvfp4_gemm_cute import DEFAULT, config_for, gemm, persisting_l2
from nvfp4_gemm_cute.testing import make_inputs

FASTCU_SCHEDULES = (
    "raster pocket-8x4 pocket-8x8 hilbert owned-plain owned-pocket-8x4 "
    "owned-pocket-8x8 hilbert-in-owned auto"
).split()


def fastcu(lib_path, inp, m, n, k, ref_out):
    lib = ctypes.CDLL(lib_path)
    lib.fastcu_gemm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    lib.fastcu_prepare.argtypes = [ctypes.c_int] * 3
    lib.fastcu_init()
    out = torch.empty_like(ref_out)

    def run():
        lib.fastcu_gemm(
            inp["a"].data_ptr(),
            inp["b"].data_ptr(),
            inp["a_sf"].data_ptr(),
            inp["b_sf"].data_ptr(),
            out.data_ptr(),
            m,
            n,
            k,
            torch.cuda.current_stream().cuda_stream,
        )

    best = None
    for s in FASTCU_SCHEDULES:
        os.environ["FASTCU_SCHEDULE"] = s
        lib.fastcu_prepare(m, n, k)
        run()
        t = time_us(run, rounds=2)[0]
        if best is None or t < best[1]:
            best = (s, t)
    os.environ["FASTCU_SCHEDULE"] = best[0]
    lib.fastcu_prepare(m, n, k)
    run()
    torch.cuda.synchronize()
    return best, torch.equal(out, ref_out), run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="4096,8192,16384", help="cube sizes or MxNxK, comma list")
    ap.add_argument("--fastcu-lib", default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument(
        "--interleaved", action="store_true", help="alternate preset and fast.cu rounds"
    )
    args = ap.parse_args()
    if args.interleaved and not args.fastcu_lib:
        ap.error("--interleaved requires --fastcu-lib")
    for s in args.shapes.split(","):
        m, n, k = (int(x) for x in s.split("x")) if "x" in s else (int(s),) * 3
        inp = make_inputs(m, n, k)
        cfg = config_for(m, n, k)
        with persisting_l2(cfg.persist_mb):
            out = gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], config=cfg)
            ours = time_us(
                lambda: gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], out=out, config=cfg),
                rounds=args.rounds,
            )
        base = time_us(
            lambda: gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], config=DEFAULT),
            rounds=args.rounds,
        )
        line = (
            f"{m}x{n}x{k}  preset {ours[0]:9.2f} us  default {base[0]:9.2f} us  "
            f"({2 * m * n * k / ours[0] / 1e6:.0f} TFLOPS)"
        )
        if args.fastcu_lib:
            (sched, _), exact, run = fastcu(args.fastcu_lib, inp, m, n, k, out)
            if args.interleaved:
                times = {"preset": [], "fast.cu": []}
                for rnd in range(args.rounds):
                    order = list(times) if rnd % 2 == 0 else list(times)[::-1]
                    for name in order:
                        fn = (
                            run
                            if name == "fast.cu"
                            else lambda: gemm(
                                inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], out=out, config=cfg
                            )
                        )
                        with persisting_l2(cfg.persist_mb if name == "preset" else 0):
                            times[name].append(time_us(fn, rounds=1)[0])
                ours = tuple(f(times["preset"]) for f in (statistics.median, min, max))
                fc = tuple(f(times["fast.cu"]) for f in (statistics.median, min, max))
                line = (
                    f"{m}x{n}x{k}  preset {ours[0]:9.2f} us "
                    f"[{ours[1]:.2f}-{ours[2]:.2f}]  default {base[0]:9.2f} us"
                )
            else:
                fc = time_us(run, rounds=args.rounds)
            line += f"  fast.cu[{sched}] {fc[0]:9.2f} us  speedup {fc[0] / ours[0]:.3f}x"
            line += "  bit-exact" if exact else "  MISMATCH"
        print(line, flush=True)


if __name__ == "__main__":
    main()
