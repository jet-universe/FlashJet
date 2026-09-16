"""flashjet: GPU (Triton/CUDA) jet reclustering for training loops."""

from .api import ALGORITHMS, ClusterOutput, cluster
from .reference import ClusterSequenceRef, cluster_event

__version__ = "0.1.0"
__all__ = [
    "cluster",
    "ClusterOutput",
    "ClusterSequenceRef",
    "cluster_event",
    "ALGORITHMS",
    "splitting_scales_from_history",
    "exclusive_jets_from_history",
    "lund_coordinates_from_history",
    "groom_from_history",
]


_HISTORY_EXPORTS = {
    "splitting_scales_from_history", "exclusive_jets_from_history",
    "lund_coordinates_from_history", "groom_from_history",
}


def __getattr__(name):
    # Keep the NumPy-only install usable without importing torch.
    if name in _HISTORY_EXPORTS:
        from . import history
        value = getattr(history, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
