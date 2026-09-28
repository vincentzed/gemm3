"""Compile the benchmark adapters against external fast.cu and ThunderKittens sources."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fastcu", type=Path, required=True, help="fast.cu checkout root")
    ap.add_argument(
        "--thunderkittens", type=Path, required=True, help="ThunderKittens checkout root"
    )
    ap.add_argument("--nvcc", default="nvcc")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    fc = args.fastcu.resolve() / "gb300/nvfp4"
    tk = args.thunderkittens.resolve()
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}

    def build(source, output, flags):
        subprocess.run(
            [
                args.nvcc,
                "-O3",
                "-shared",
                "-Xcompiler=-fPIC",
                *flags,
                str(here / source),
                "-o",
                str(here / output),
            ],
            env=env,
            check=True,
        )

    build("cublaslt_lib.cu", "libcublaslt_bench.so", ["-std=c++17", "-lcublasLt"])
    flags = [
        "-std=c++17",
        "-DNDEBUG",
        "-gencode=arch=compute_103a,code=sm_103a",
        "-lcublasLt",
        "-lcuda",
    ]
    build("fastcu_lib.cu", "libfastcu_latest.so", [*flags, f"-I{fc}"])
    # The upstream route table covers 4096 output tiles. 32K needs 16384.
    # Its two enlarged int32 tables exceed constant memory, so use global memory
    # for that shape only. Keep the upstream source and <=16K binary unchanged.
    with tempfile.TemporaryDirectory(prefix="fastcu-32k-") as work:
        expanded = Path(work) / "nvfp4"
        shutil.copytree(fc, expanded)
        header = expanded / "gemm9.cuh"
        text = header.read_text()
        old = "constexpr int L2A_ROUTE_WORK_CAP = 64 * 64;"
        old_table = "__device__ __constant__ int l2a_route_tables["
        if text.count(old) != 1 or text.count(old_table) != 1:
            raise RuntimeError("Upstream schedule storage changed; review the 32K adapter")
        header.write_text(
            text.replace(old, "constexpr int L2A_ROUTE_WORK_CAP = 128 * 128;").replace(
                old_table, "__device__ int l2a_route_tables["
            )
        )
        build("fastcu_lib.cu", "libfastcu_latest_32k.so", [*flags, f"-I{expanded}"])
    build(
        "thunderkittens_lib.cu",
        "libtk_bench.so",
        [
            "-std=c++20",
            "--use_fast_math",
            "-Xcompiler=-fno-strict-aliasing",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            "-DKITTENS_SM103",
            "-gencode=arch=compute_103a,code=sm_103a",
            "-lcuda",
            f"-I{tk}",
            f"-I{tk / 'include'}",
            f"-I{tk / 'prototype'}",
        ],
    )


if __name__ == "__main__":
    main()
