"""Render measured GEMM throughput, following compact kernel-paper figure layouts.

uv run --no-project --with matplotlib==3.10.8 python figures/plot.py
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
LIBRARIES = [
    ("preset", "GEMM³ (Us)", "#8064A2"),
    ("fastcu", "fast.cu · best of 9", "#4878A8"),
    ("cublaslt", "cuBLASLt", "#6D9E9A"),
    ("cutlass", "CUTLASS", "#6D9E9A"),
    ("cudnn", "cuDNN", "#6D9E9A"),
    ("cute-dsl", "FlashInfer CuTe DSL", "#7D93AD"),
    ("quack", "QuACK", "#7D93AD"),
    ("thunderkittens", "ThunderKittens †", "#CB9872"),
]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": "#D8DCE1",
            "grid.linewidth": 0.6,
            "grid.linestyle": (0, (2, 3)),
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "svg.hashsalt": "nvfp4-gemm-cute",
            "savefig.facecolor": "white",
        }
    )


def throughput(size, latency_us):
    """Dense GEMM work: one multiply-add counts as two FLOPs; time is in µs."""
    return 2 * size**3 / (latency_us * 1e6)


def save(fig, name, formats=("svg", "pdf", "png")):
    metadata = {"Creator": "nvfp4-gemm-cute / figures/plot.py"}
    for ext in formats:
        extra = {"Date": None} if ext == "svg" else {}
        if ext == "pdf":
            extra = {"CreationDate": None, "ModDate": None}
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, metadata=metadata | extra)
        if ext == "svg":
            path = OUT / f"{name}.{ext}"
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def library_panel(ax, data, limit):
    size = data["size"]
    for y, (key, label, color) in enumerate(LIBRARIES):
        if key not in data["results"]:
            ax.text(0.15, y, "unavailable", va="center", fontsize=8, color="#777777")
            continue
        samples = data["results"][key]["samples_us"]
        value = throughput(size, statistics.median(samples)) / 1000
        lo, hi = (throughput(size, f(samples)) / 1000 for f in (max, min))
        ax.barh(y, value, height=0.62, color=color, zorder=3)
        ax.errorbar(
            value,
            y,
            xerr=[[value - lo], [hi - value]],
            fmt="none",
            ecolor="#30343A",
            elinewidth=0.75,
            capsize=2,
            capthick=0.75,
            zorder=4,
        )
        ax.text(
            hi + 0.12,
            y,
            f"{value:.2f}",
            va="center",
            fontsize=8,
            fontweight="bold" if key == "preset" else "normal",
        )
    ax.set_title(f"{size:,} × {size:,} × {size:,}", loc="left", pad=9)
    ax.set_yticks(range(len(LIBRARIES)), [row[1] for row in LIBRARIES])
    ax.tick_params(axis="y", length=0, pad=7, labelsize=8.4)
    ax.set_ylim(len(LIBRARIES) - 0.4, -0.7)
    ax.set_xlim(0, limit)
    ax.set_xticks(range(0, limit + 1, 2))
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    ax.tick_params(axis="x", labelbottom=True)


def axis_limit(datasets):
    maximum = max(
        throughput(d["size"], min(v["samples_us"])) / 1000
        for d in datasets
        for v in d["results"].values()
    )
    return max(10, int(maximum + 1.7))


def libraries(datasets):
    """Small multiples retain readable library names and a shared, zero-based axis."""
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.1), sharex=True)
    fig.subplots_adjust(left=0.195, right=0.97, bottom=0.15, top=0.90, wspace=0.72, hspace=0.26)
    limit = axis_limit(datasets)
    for ax, data in zip(axes.flat, datasets, strict=True):
        library_panel(ax, data, limit)
    for ax in axes[1]:
        ax.set_xlabel("Throughput (PFLOP/s)", labelpad=7)
    fig.suptitle("NVFP4 GEMM on NVIDIA B300", fontsize=13, fontweight="medium", y=0.98)
    fig.text(
        0.195,
        0.035,
        "† BF16 output; other rows FP16. ThunderKittens pads K to a multiple of 768.\n"
        "CUPTI · CUDA graphs · cold L2",
        fontsize=8,
        color="#555555",
        linespacing=1.6,
    )
    save(fig, "performance")


def single_shape(data):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.23, top=0.84)
    library_panel(ax, data, axis_limit([data]))
    ax.set_xlabel("Throughput (PFLOP/s)", labelpad=7)
    fig.suptitle("NVFP4 GEMM on NVIDIA B300", fontsize=13, fontweight="medium", y=0.97)
    fig.text(
        0.035,
        0.035,
        "† BF16 output; other rows FP16. ThunderKittens pads K to a multiple of 768.\n"
        "CUPTI · CUDA graphs · cold L2",
        fontsize=8,
        color="#555555",
        linespacing=1.6,
    )
    save(fig, f"performance_{data['size'] // 1024}k", formats=("svg", "png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size", type=int, choices=(4096, 8192, 16384, 32768), help="export one shape as SVG/PNG"
    )
    args = parser.parse_args()
    style()
    sizes = (args.size,) if args.size else (4096, 8192, 16384, 32768)
    paths = [ROOT / f"benchmarks/results/libraries-{size}.json" for size in sizes]
    datasets = [json.loads(path.read_text()) for path in paths]
    assert len({data["gpu_uuid"] for data in datasets}) == 1
    for data in datasets:
        assert not data["errors"], data["errors"]
        assert set(data["results"]) == {row[0] for row in LIBRARIES}
        for result in data["results"].values():
            assert len(result["samples_us"]) == data["rounds"]
            assert result["median_us"] == statistics.median(result["samples_us"])
    if args.size:
        single_shape(datasets[0])
        print(f"Wrote performance_{args.size // 1024}k.svg and 300 dpi PNG.")
    else:
        libraries(datasets)
        print("Wrote throughput figures in SVG, PDF and 300 dpi PNG.")


if __name__ == "__main__":
    main()
