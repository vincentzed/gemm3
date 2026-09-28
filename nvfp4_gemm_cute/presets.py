"""Per-shape launch configurations measured on B300 (sm_103).

A configuration fixes the MMA tile, the preferred cluster, the fallback cluster for SMs that
cannot host the preferred one, and kernel attributes (tile order, L2 cache hints, prefetch).
``persist_mb`` selects the host persisting-L2 limit used with the preset's operand
eviction hints (see ``api.persisting_l2``). The limit and hints do not guarantee residency.

Timings and measurement conditions are recorded in ``benchmarks/results/libraries-*.json``.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Launch configuration for one GEMM shape."""

    tile: tuple = (256, 256)
    cluster: tuple = (4, 1)
    fallback: tuple | None = (2, 1)
    attrs: tuple = field(default_factory=tuple)  # sorted ((name, value), ...)
    persist_mb: int = 0

    @staticmethod
    def make(tile=(256, 256), cluster=(4, 1), fallback=(2, 1), persist_mb=0, **attrs):
        return Config(
            tuple(tile),
            tuple(cluster),
            fallback and tuple(fallback),
            tuple(sorted(attrs.items())),
            persist_mb,
        )

    def attr_dict(self):
        return dict(self.attrs)


# Mixed 4x1 clusters with 2x1 fallback, plain raster: the general-purpose default.
DEFAULT = Config.make()

# (M, N, K) -> Config.
PRESETS = {
    # Two-stripe N sweep and initial operand prefetch for the short 4K mainloop.
    (4096, 4096, 4096): Config.make(
        cluster=(4, 2),
        fallback=(2, 1),
        fast_swz=2,
        raster_along_m=False,
        epi_store_pace_per_ktile_ns=50,
        prefetch_first_ktiles=1,
    ),
    # Exact nine-slot ownership map with A/SFA retention and a 48 MiB L2 limit.
    (8192, 8192, 8192): Config.make(
        cluster=(8, 2),
        fallback=(2, 1),
        persist_mb=48,
        fast_swz=4,
        raster_along_m=False,
        owned_8k=True,
        l2_policy_a="evict_last",
        l2_policy_sfa="evict_last",
        epi_store_pace_per_ktile_ns=50,
    ),
    # Short-tail pipeline and shared-memory MMA descriptor templates; 40 MiB L2 limit.
    (16384, 16384, 16384): Config.make(
        cluster=(8, 2),
        fallback=(2, 1),
        persist_mb=40,
        fast_swz=4,
        raster_along_m=False,
        l2_policy_a="evict_last",
        l2_policy_sfa="evict_last",
        trim_short_tail=True,
        tma_b_first=True,
        tma_sfb_first=True,
        retire_a_cols=8,
        packed_ab_desc=True,
    ),
    # 4x4 reuse geometry, compact descriptor issue, and fixed-shape compilation.
    # Needs a 48 MiB persisting-L2 set-aside.
    (32768, 32768, 32768): Config.make(
        cluster=(4, 4),
        fallback=(2, 1),
        persist_mb=48,
        fast_swz=4,
        raster_along_m=False,
        l2_policy_a="evict_last",
        epi_store_pair=True,
        epi_store_pace_per_ktile_ns=0,
        retire_a_cols=8,
        packed_ab_desc=True,
        specialize_32k=True,
    ),
}


def config_for(m: int, n: int, k: int) -> Config:
    """Return the tuned configuration for (m, n, k), or DEFAULT."""
    return PRESETS.get((m, n, k), DEFAULT)
