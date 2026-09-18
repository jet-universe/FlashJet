"""Serve FlashJet as an inference service.

Clustering is not inference, but a Triton Inference Server is a convenient way
to give a GPU to processes that cannot use one directly -- a reconstruction
framework that hands a module one event at a time, or jobs spread over a batch
farm.  The server then supplies what FlashJet needs and a single caller cannot:
requests that arrive together are clustered in one kernel launch.

:func:`cluster_events` is the piece that does the work, and is independent of
Triton: a list of events in, a particle -> jet map per event out.  It runs the
Triton kernels when torch and a GPU are there, FlashJet's C++ kernel when they
are not, and the NumPy reference as a last resort.

:mod:`flashjet.serving.triton_model` wraps it as a Triton Python-backend model,
and :mod:`flashjet.serving.model_repository` writes a model repository that
uses it.
"""

import numpy as np

MAX_PARTICLES = 16384  # the large-N Triton kernel's limit, see triton_large.py

__all__ = ["MAX_PARTICLES", "resolve_backend", "cluster_events"]


def resolve_backend(choice="auto"):
    """The backend :func:`cluster_events` would use.

    ``"auto"`` prefers the GPU, then the compiled C++ kernel, then NumPy.  Any
    other choice ("gpu", "native", "numpy") is checked and returned, so a server
    fails at start-up rather than on the first request.
    """
    if choice in ("auto", "gpu"):
        try:
            import torch

            if torch.cuda.is_available():
                return "gpu"
        except ImportError:
            pass
        if choice == "gpu":
            raise RuntimeError("flashjet.serving: backend 'gpu' needs torch with CUDA")
    if choice in ("auto", "native"):
        from .. import _native

        if _native.HAS_NATIVE:
            return "native"
        if choice == "native":
            raise RuntimeError("flashjet.serving: the C++ kernel (_flashjet_cpu*.so) is not built")
    if choice not in ("auto", "numpy"):
        raise ValueError(f"flashjet.serving: unknown backend {choice!r}")
    return "numpy"


def _pad(events):
    """Pad a list of (n_i, 4) events into one (B, N, 4) batch and its mask."""
    width = max((len(x) for x in events), default=0)
    batch = np.zeros((len(events), max(width, 1), 4), dtype=np.float64)
    mask = np.zeros((len(events), max(width, 1)), dtype=bool)
    for b, event in enumerate(events):
        batch[b, : len(event)] = event
        mask[b, : len(event)] = True
    return batch, mask


def cluster_events(events, R=0.4, p=-1.0, backend="auto", threads=1):
    """Cluster several events in one call.

    Args:
        events: list of (n_i, 4) arrays, columns (px, py, pz, E).  The events
            are padded to a common width, which is where the GPU kernels get
            their parallelism from, so passing many small events at once is the
            point of this function.
        R, p: jet radius and generalized-kt exponent (-1 anti-kt, 0 C/A, 1 kt).
        backend: see :func:`resolve_backend`.
        threads: worker threads for the C++ kernel.
    Returns:
        list of ``(jet_idx, n_jets)``, one per event: ``jet_idx`` is an (n_i,)
        int32 array of jet indices in beam-merge order.
    """
    if not events:
        return []
    widest = max(len(x) for x in events)
    if widest > MAX_PARTICLES:
        raise ValueError(
            f"event with {widest} particles exceeds the {MAX_PARTICLES} the GPU kernels support; "
            "cluster it on the CPU instead"
        )
    chosen = resolve_backend(backend) if backend in ("auto", "gpu", "native", "numpy") else backend

    if chosen == "gpu":
        import torch

        from .. import cluster

        batch, mask = _pad(events)
        out = cluster(torch.from_numpy(batch).cuda(), torch.from_numpy(mask).cuda(), R=R, p=p, validate=False)
        jet_idx = out.jet_idx.cpu().numpy()
        n_jets = out.n_jets.cpu().numpy()
        return [(jet_idx[b, : len(x)].astype(np.int32), int(n_jets[b])) for b, x in enumerate(events)]

    if chosen == "native":
        from .. import _native
        from ..history_np import jet_idx_from_history_np

        batch, mask = _pad(events)
        hp1, hp2, hch, _ = _native.cluster_native(batch, mask.view(np.uint8), R, p, threads)
        return [jet_idx_from_history_np(hp1[b], hp2[b], hch[b], len(x)) for b, x in enumerate(events)]

    from ..nn_reference import cluster_event_nn

    out = []
    for event in events:
        result = cluster_event_nn(event, R=R, p=p)
        out.append((np.asarray(result["jet_idx"], dtype=np.int32), int(result["n_jets"])))
    return out
