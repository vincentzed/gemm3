"""Query device properties the kernel is specialized on at compile time."""

import functools

import torch


@functools.cache
def get_max_active_clusters(cluster_size: int) -> int:
    """Return how many clusters of cluster_size CTAs can be co-resident on the current GPU.

    Uses the CuTe DSL hardware probe. If the probe fails (it needs a current driver context),
    falls back to the SM count divided by the cluster size.
    """
    torch.cuda.init()
    try:
        from cutlass.utils import HardwareInfo

        return HardwareInfo().get_max_active_clusters(cluster_size)
    except Exception:
        sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        return sms // cluster_size


def arch() -> str:
    """Return the compute architecture of the current CUDA device, such as "sm_103"."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"
