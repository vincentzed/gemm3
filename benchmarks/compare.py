"""Compare source-built NVFP4 kernels on identical quantized inputs.

See docs/benchmarks.md for dependency revisions and wrapper build commands.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from importlib.metadata import version

import torch

from harness import time_us
from bench import FASTCU_SCHEDULES
from nvfp4_gemm_cute import config_for, gemm, persisting_l2
from nvfp4_gemm_cute.quant import dequantize_nvfp4, swizzle_sf
from nvfp4_gemm_cute.testing import make_inputs


def shared(name, pointers):
    lib = ctypes.CDLL(str(Path(__file__).parent / name))
    for symbol, count in pointers.items():
        fn = getattr(lib, symbol)
        fn.argtypes = [ctypes.c_void_p] * count
    return lib


def adapter(name, inp, size):
    """Return run, output, metadata, keepalive, cleanup, L2 reservation (MiB)."""
    m = n = k = size
    bf16 = name == "thunderkittens"
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16 if bf16 else torch.float16)
    stream = lambda: torch.cuda.current_stream().cuda_stream
    meta, keep, cleanup, persist = {}, [], lambda: None, 0
    if name == "preset":
        cfg = config_for(m, n, k)
        persist = cfg.persist_mb
        run = lambda: gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], out=out, config=cfg)
    elif name == "cublaslt":
        lib = shared("libcublaslt_bench.so", {"cublas_run": 2, "cublas_destroy": 1})
        lib.cublas_create.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
        lib.cublas_create.restype = ctypes.c_void_p
        ctx = lib.cublas_create(
            *(inp[x].data_ptr() for x in ("a", "b", "a_sf", "b_sf")), out.data_ptr(), m, n, k
        )
        if not ctx:
            raise RuntimeError("cuBLASLt plan creation failed")
        lib.cublas_workspace.argtypes = [ctypes.c_void_p]
        lib.cublas_workspace.restype = ctypes.c_size_t
        meta.update(version=lib.cublas_version(), workspace_bytes=lib.cublas_workspace(ctx))
        run = lambda: checked(lib.cublas_run(ctx, stream()))
        cleanup = lambda: lib.cublas_destroy(ctx)
    elif name == "fastcu":
        suffix = "_32k" if size == 32768 else ""
        lib = shared(f"libfastcu_latest{suffix}.so", {})
        lib.fastcu_gemm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.fastcu_prepare.argtypes = [ctypes.c_int] * 3
        lib.fastcu_init()
        run = lambda: checked(
            lib.fastcu_gemm(
                *(inp[x].data_ptr() for x in ("a", "b", "a_sf", "b_sf")),
                out.data_ptr(),
                m,
                n,
                k,
                stream(),
            )
        )
        scores = {}
        for schedule in FASTCU_SCHEDULES:
            os.environ["FASTCU_SCHEDULE"] = schedule
            lib.fastcu_prepare(m, n, k)
            run()
            scores[schedule] = time_us(run, rounds=2)[0]
        best = min(scores, key=scores.get)
        os.environ["FASTCU_SCHEDULE"] = best
        lib.fastcu_prepare(m, n, k)
        meta.update(schedule=best, schedule_screen_us=scores)
    elif name == "thunderkittens":
        kp = (k + 767) // 768 * 768
        padded = {}
        for x in ("a", "b"):
            padded[x] = torch.nn.functional.pad(inp[x], (0, (kp - k) // 2))
            sf = torch.nn.functional.pad(inp[x + "_sf_linear"], (0, (kp - k) // 16))
            padded[x + "_sf"] = swizzle_sf(sf)
        one = torch.ones((), device="cuda")
        lib = shared("libtk_bench.so", {"tk_run": 2, "tk_destroy": 1})
        lib.tk_create.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
        lib.tk_create.restype = ctypes.c_void_p
        ctx = lib.tk_create(
            *(padded[x].data_ptr() for x in ("a", "b", "a_sf", "b_sf")),
            out.data_ptr(),
            one.data_ptr(),
            m,
            n,
            kp,
        )
        if not ctx:
            raise RuntimeError("ThunderKittens launch setup failed")
        run = lambda: checked(lib.tk_run(ctx, stream()))
        cleanup = lambda: lib.tk_destroy(ctx)
        keep = [padded, one]
        meta["padded_k"] = kp
    elif name == "quack":
        from quack.blockscaled.operand import BlockScaledOperand
        from quack.gemm_interface import gemm as qgemm

        operands = [
            BlockScaledOperand.from_parts(
                inp[x].view(torch.float4_e2m1fn_x2),
                inp[x + "_sf"].view(torch.float8_e4m3fn).reshape(size // 128, k // 64, 32, 4, 4),
                "nvfp4",
            )
            for x in ("a", "b")
        ]
        run = lambda: qgemm(operands[0], operands[1].mT, out=out, out_dtype=out.dtype, tuned=True)
        keep = operands
        meta["tuned"] = True
    elif name in ("cutlass", "cudnn", "cute-dsl"):
        import flashinfer

        b, bsf = inp["b"], inp["b_sf"].reshape(n, k // 16)
        alpha = torch.ones((), device="cuda")
        run = lambda: flashinfer.mm_fp4(
            inp["a"],
            b.T,
            inp["a_sf"].reshape(m, k // 16),
            bsf.T,
            alpha,
            out_dtype=out.dtype,
            out=out,
            backend=name,
        )
        keep = [b, bsf, alpha]
        with flashinfer.autotune(tune_mode=True):
            run()
        meta["autotuned"] = True
    else:
        raise ValueError(f"unsupported benchmark implementation: {name}")
    return run, out, meta, keep, cleanup, persist


def checked(code):
    if code:
        raise RuntimeError(f"CUDA/library launch returned {code}")


def accuracy(out, ref):
    # Chunk the reduction so a 32K output does not require several GiB of temporaries.
    sqerr = torch.zeros((), device="cuda", dtype=torch.float64)
    sqref = torch.zeros_like(sqerr)
    exact = True
    finite = True
    for start in range(0, out.shape[0], 128):
        a, b = out[start : start + 128].float(), ref[start : start + 128].float()
        sqerr += (a - b).square().sum(dtype=torch.float64)
        sqref += b.square().sum(dtype=torch.float64)
        exact = exact and torch.equal(a, b)
        finite = finite and bool(torch.isfinite(a).all())
    rel = (sqerr / sqref).sqrt().item()
    if not finite or rel > (0.003 if out.dtype == torch.bfloat16 else 0.0005):
        raise RuntimeError(f"Numerical check failed: relative L2={rel}, finite={finite}")
    return dict(bit_exact=exact, relative_l2=rel, finite=finite)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument(
        "--only",
        default="preset,fastcu,cublaslt,thunderkittens,quack,cutlass,cudnn,cute-dsl",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    inp = make_inputs(args.size, args.size, args.size)
    cfg = config_for(args.size, args.size, args.size)
    with persisting_l2(cfg.persist_mb):
        refs = {
            dtype: gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], out_dtype=dtype, config=cfg)
            for dtype in (torch.float16, torch.bfloat16)
        }
    # Independent FP32 check at rows/columns spread across the full output.
    indices = torch.linspace(0, args.size - 1, 128, device="cuda").long()
    a = dequantize_nvfp4(inp["a"][indices], inp["a_sf_linear"][indices])
    b = dequantize_nvfp4(inp["b"][indices], inp["b_sf_linear"][indices])
    torch.backends.cuda.matmul.allow_tf32 = False
    sample_ref = a @ b.T
    reference_checks = {
        str(dtype): accuracy(out[indices][:, indices], sample_ref) for dtype, out in refs.items()
    }
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")

    def compute_pids():
        return subprocess.check_output(
            ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).split()

    expected_pids = compute_pids()
    if len(expected_pids) != 1:
        raise RuntimeError(f"Expected exclusive GPU use, found processes {expected_pids}")

    def check_exclusive():
        actual = compute_pids()
        if actual != expected_pids:
            raise RuntimeError(f"GPU process set changed to {actual}; discard this run")
        return actual

    source_root = Path(__file__).resolve().parents[1]
    data = dict(
        size=args.size,
        rounds=args.rounds,
        date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=uuid,
        torch=torch.__version__,
        cupti_python=version("cupti-python"),
        timing="CUPTI; CUDA graph; cold L2; cooldown; 50 samples/round",
        reference_checks=reference_checks,
        results={},
        errors={},
        gpu_checks=[],
        config=dict(
            cluster=cfg.cluster,
            fallback=cfg.fallback,
            persist_mb=cfg.persist_mb,
            attrs=cfg.attr_dict(),
        ),
        source_sha256={
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                *(
                    source_root / "nvfp4_gemm_cute" / name
                    for name in ("kernel.py", "epilogue.py", "api.py", "presets.py")
                ),
                Path(__file__).resolve(),
                Path(__file__).with_name("harness.py").resolve(),
            ]
        },
    )
    runners = {}

    def save():
        data["loaded_libraries"] = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if any(x in line for x in ("libcublas", "libcudnn", "libcupti", "libnvrtc"))
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")

    for name in args.only.split(","):
        check_exclusive()
        print(f"Preparing {name} at {args.size}", flush=True)
        try:
            item = adapter(name, inp, args.size)
            run, out, meta, keep, cleanup, persist = item
            with persisting_l2(persist):
                run()
                torch.cuda.synchronize()
            meta.update(
                dtype=str(out.dtype), correctness=accuracy(out, refs[out.dtype]), samples_us=[]
            )
            data["results"][name] = meta
            runners[name] = item
            print(f"Ready {name}: {meta}", flush=True)
        except Exception as exc:
            data["errors"][name] = repr(exc)
            print(f"FAILED {name}: {exc}", flush=True)
        save()
    names = list(runners)
    if not names:
        raise RuntimeError("No implementation passed validation")
    for rnd in range(args.rounds):
        processes = check_exclusive()
        data["gpu_checks"].append(dict(round=rnd + 1, compute_pids=processes))
        order = names[rnd % len(names) :] + names[: rnd % len(names)]
        if rnd % 2:
            order = order[::-1]
        for name in order:
            check_exclusive()
            run, out, meta, keep, cleanup, persist = runners[name]
            with persisting_l2(persist):
                us = time_us(run, rounds=1)[0]
            meta["samples_us"].append(us)
            meta["median_us"] = statistics.median(meta["samples_us"])
            meta["tflops"] = 2 * args.size**3 / meta["median_us"] / 1e6
            print(f"Round {rnd + 1} {name}: {us:.3f} us", flush=True)
            check_exclusive()
            save()
    for item in runners.values():
        item[4]()


if __name__ == "__main__":
    main()
