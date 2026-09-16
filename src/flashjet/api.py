"""Public API: flashjet.cluster().

Designed for use inside training loops: inputs are padded torch tensors that
stay on the GPU, the clustering returns a particle->jet assignment plus the
full merge history, and jet four-momenta are recovered with a differentiable
scatter_add (exact for E-scheme recombination).
"""

from dataclasses import dataclass

import numpy as np

ALGORITHMS = {
    "antikt": -1.0,
    "anti-kt": -1.0,
    "kt": 1.0,
    "cambridge": 0.0,
    "ca": 0.0,
    "cambridge-aachen": 0.0,
}


@dataclass
class ClusterOutput:
    """Batched clustering result (all tensors live on the input device).

    Attributes:
        jet_idx:    (B, N) int64, jet index per input particle in beam-merge
                    order, -1 for padding (or particles below ptmin filters
                    applied later).
        n_jets:     (B,) int64.
        hist_p1/p2/child: (B, N) int64 merge tree (pseudojet ids; initial
                    particles are 0..n-1 in mask order, -1 = beam, -2 = pad).
        hist_d:     (B, N) float, d_min of each merge step.
        mask:       (B, N) bool, True for real particles (needed by the
                    exclusive-jet / substructure decoders that map slots ->
                    initial pseudojet ids).  None if the output was built
                    without it (older callers); the exclusive/grooming helpers
                    then require an explicit mask=.
    """

    jet_idx: "object"
    n_jets: "object"
    hist_p1: "object"
    hist_p2: "object"
    hist_child: "object"
    hist_d: "object"
    mask: "object" = None

    def jets_p4(self, p4, n_jets_max=None):
        """Differentiable jet four-momenta via scatter_add of constituents.

        Args:
            p4: the (B, N, 4) input tensor (may require grad).
        Returns:
            (B, J, 4) tensor, J = n_jets.max() (or n_jets_max), zero-padded,
            jets in beam-merge order (use sort_jets_by_pt to reorder).
            Constituents of jets >= n_jets_max are dropped, not folded into
            the last slot.

        Note:
            Recovery uses scatter_add, whose CUDA kernel accumulates in a
            nondeterministic order: unlike the clustering itself (the kernels
            are bitwise-deterministic), the returned momenta can vary run to
            run at the ulp level.  Enable torch.use_deterministic_algorithms(True)
            if you need bitwise-stable jet four-momenta.
        """
        import torch

        if self.jet_idx is None:
            raise ValueError(
                "jets_p4 needs the per-particle decode, but this output was "
                "produced with decode=False.  Re-run cluster(..., decode=True), "
                "or use splitting_scales() for history-only substructure features."
            )
        B, N, _ = p4.shape
        J = int(self.n_jets.max().item()) if n_jets_max is None else n_jets_max
        J = max(J, 1)
        out = p4.new_zeros(B, J, 4)
        valid = (self.jet_idx >= 0) & (self.jet_idx < J)
        idx = self.jet_idx.clamp(0, J - 1)
        src = torch.where(valid.unsqueeze(-1), p4, torch.zeros_like(p4))
        out.scatter_add_(1, idx.unsqueeze(-1).expand(B, N, 4), src)
        return out

    def sort_jets_by_pt(self, jets_p4):
        """Return (sorted_jets, order) with jets sorted pt-descending per event."""
        import torch

        pt = torch.hypot(jets_p4[..., 0], jets_p4[..., 1])
        order = pt.argsort(dim=1, descending=True, stable=True)
        return torch.take_along_dim(jets_p4, order.unsqueeze(-1), dim=1), order

    def splitting_scales(self, n_jets_max=None):
        """Per-jet sequential-recombination splitting scales (B, J, S) float.

        out[:, j, 0] is jet j's last merge / first de-clustering split (the
        d_12 scale), [:, j, 1] the next (d_23), ..., zero-padded past each
        jet's merge count; jets are in beam-merge order, ALIGNED with jets_p4
        (so per-jet features concatenate -- if you sort_jets_by_pt the jets,
        gather these with the same `order`).  For the kt algorithm these are
        the exclusive d_12 >= d_23 >= ... scales (kt-splitting / Lund inputs);
        for C/A and anti-kt the entries are the de-clustering sequence but not
        value-sorted.  See history.splitting_scales_from_history for details.

        Reads only the merge history, so it works whether or not the
        per-particle decode ran (cluster(..., decode=False) is fine).
        """
        from .history import splitting_scales_from_history

        return splitting_scales_from_history(
            self.hist_p1, self.hist_p2, self.hist_child, self.hist_d, n_jets_max
        )

    def _require_mask(self, mask):
        if mask is None:
            mask = self.mask
        if mask is None:
            raise ValueError(
                "this decoder needs the event mask (slot -> initial-id map); "
                "this ClusterOutput was built without one -- pass mask=."
            )
        return mask

    def exclusive_jets(self, n_jets=None, d_cut=None, mask=None):
        """Exclusive-jet particle assignment (undo the sequence's last merges).

        Pass exactly one of `n_jets` (leave this many jets) or `d_cut` (undo
        every pair-merge with d >= d_cut, the exclusive-y_cut form on the stored
        d = min(w_i, w_j) * dR^2 / R^2).  Returns (excl_idx (B, N) int64,
        n_excl (B,)); excl_idx feeds jets_p4() directly.  This is the classic
        *kt exclusive* mode; on kt (p=1) d is monotonic so both cuts are exact.
        """
        from .history import exclusive_jets_from_history

        mask = self._require_mask(mask)
        return exclusive_jets_from_history(
            self.hist_p1, self.hist_p2, self.hist_child, self.hist_d, mask,
            n_jets=n_jets, d_cut=d_cut,
        )

    def lund_coordinates(self, p4, R, n_jets_max=None, mask=None):
        """Per-jet, per-split Lund-plane coordinates (B, J, S, 6).

        Channels: (z, dR, kt, ln 1/dR, ln kt, d), one row per de-clustering
        split in the same order as splitting_scales() (slot 0 = widest split),
        aligned with jets_p4().  Extends splitting_scales with the full Lund
        inputs; see history.lund_coordinates_from_history.
        """
        from .history import lund_coordinates_from_history

        mask = self._require_mask(mask)
        return lund_coordinates_from_history(
            self.hist_p1, self.hist_p2, self.hist_child, self.hist_d, mask,
            p4, R, n_jets_max,
        )

    def groomed_jets(self, p4, R, z_cut=0.1, beta=0.0, mu=None,
                     n_jets_max=None, mask=None):
        """Soft-drop / mass-drop grooming of each jet (declustering tagger).

        Walks each jet down the harder branch, dropping soft wide-angle
        radiation until a split satisfies z > z_cut*(dR/R)**beta (beta=0 is
        mMDT); pass mu= to also require max(m_i,m_j) < mu*m (Mass-Drop Tagger).
        Returns a dict with groomed_p4 (B, J, 4), tagged (B, J) bool, and the
        passing split's z / dR / mu_split / n_drop; jets in beam-merge order
        aligned with jets_p4().  See history.groom_from_history.
        """
        from .history import groom_from_history

        mask = self._require_mask(mask)
        return groom_from_history(
            self.hist_p1, self.hist_p2, self.hist_child, self.hist_d, mask,
            p4, R, z_cut=z_cut, beta=beta, mu=mu, n_jets_max=n_jets_max,
        )

    def mass_drop(self, p4, R, mu=0.67, y_cut=0.09, n_jets_max=None, mask=None):
        """Original Mass-Drop Tagger (Butterworth-Davison-Rubin-Salam).

        Convenience wrapper over groomed_jets with the mass-drop mu and a
        z_cut derived from y_cut (z_cut = y_cut, beta = 0); returns the same
        dict.  Use groomed_jets(..., beta, mu) for the general soft-drop form.
        """
        return self.groomed_jets(
            p4, R, z_cut=y_cut, beta=0.0, mu=mu,
            n_jets_max=n_jets_max, mask=mask,
        )


def cluster(p4, mask=None, R=0.4, algorithm="antikt", p=None, backend="auto", validate=True, decode=True):
    """Cluster particles with a generalized-kt sequential recombination.

    Args:
        p4: (n, 4) numpy array for a single event (returns the NumPy
            reference ClusterSequenceRef), or a (B, N, 4) torch tensor
            (px, py, pz, E) for a padded batch.
        mask: (B, N) bool tensor for the batched path (default: all true).
        R: jet radius.
        algorithm: 'antikt' | 'kt' | 'cambridge' (ignored if p given).
        p: generalized-kt exponent overriding `algorithm`.
        backend: 'auto' | 'triton' | 'triton-large' | 'cpu' | 'torch'.
            'auto' picks the fused register kernel (CUDA, N <= 16, where it
            measures fastest), then the scratch NN-array kernel (CUDA,
            N <= 16384), then the compiled C++ CPU kernel for CPU tensors,
            then torch.  Set FLASHJET_TUNE=1 to let the triton-large backend
            autotune its launch params once per GPU model (see
            flashjet/tune.py; persisted, reproducible afterwards), and
            FLASHJET_COMPILE_DECODE=1 to torch.compile its history decode
            (see history.py; helps large direct calls at small N).
        validate: reject non-finite four-momenta in masked-in slots up front
            (a single inf row can otherwise abort the CUDA context in the
            decode's scatter).  Costs one device sync per call; pass False
            in hot loops with trusted inputs.
        decode: triton-large only -- when False, skip the per-particle
            jet_idx pointer-jump decode (ClusterOutput.jet_idx is then None)
            for callers that only need the merge history / substructure
            features (ClusterOutput.splitting_scales).  Other backends compute
            jet_idx in-kernel and ignore this flag.

    Returns:
        ClusterSequenceRef (single event) or ClusterOutput (batch).
    """
    if p is None:
        try:
            p = ALGORITHMS[algorithm.lower()]
        except KeyError:
            raise ValueError(f"unknown algorithm {algorithm!r}; use one of {sorted(set(ALGORITHMS))} or pass p=")

    # R must be positive: at R == 0 the 1/(R*R) the kernels pass is a host-side
    # ZeroDivisionError (triton paths) or a silent inf (torch/numpy), and R < 0
    # is otherwise used unchecked as |R|.  Guard once here for every backend.
    if not R > 0:  # NaN-safe: `not nan > 0` is True
        raise ValueError(f"R must be positive, got {R!r}")

    if isinstance(p4, np.ndarray):
        from .reference import cluster_event

        if validate and not np.isfinite(p4).all():
            raise ValueError("p4 contains non-finite values")
        return cluster_event(p4, R=R, p=p)

    import torch

    if not isinstance(p4, torch.Tensor) or p4.ndim != 3 or p4.shape[-1] != 4:
        raise TypeError("p4 must be a (n,4) numpy array or a (B,N,4) torch tensor")
    if mask is None:
        mask = torch.ones(p4.shape[:2], dtype=torch.bool, device=p4.device)
    if not isinstance(mask, torch.Tensor) or mask.shape != p4.shape[:2] or mask.dtype != torch.bool:
        raise TypeError("mask must be a (B, N) bool tensor")
    if mask.device != p4.device:
        raise ValueError(f"mask must be on the same device as p4 (mask: {mask.device}, p4: {p4.device})")
    if validate and not bool((torch.isfinite(p4) | ~mask.unsqueeze(-1)).all()):
        raise ValueError("p4 contains non-finite values in masked-in slots (pass validate=False to skip this check)")

    from .triton_backend import HAS_TRITON, MAX_BLOCK
    from .triton_large import MAX_LARGE_N

    N = p4.shape[1]
    chosen = backend
    if backend == "auto":
        # A100 re-measure (B in 256..8192, 2026-06): triton-large is faster than
        # the fused O(N^3) register kernel at all N >= ~20 (up to 2.2x at N=32,
        # where fused's dense N*N tile dominates); fused only wins marginally
        # (<= 8%) at N <= 16 on large batches.  So the crossover is N <= 16 --
        # NOT the 32-64 a pre-roadmap baseline sweep had estimated, which routed
        # the whole N=17..32 band to the slower kernel.
        if HAS_TRITON and p4.is_cuda and N <= 16:
            chosen = "triton"
        elif HAS_TRITON and p4.is_cuda and N <= MAX_LARGE_N:
            chosen = "triton-large"
        elif not p4.is_cuda:
            # O(N^2) NN strategy; the torch backend is O(N^3) and only wins on
            # CUDA, where the cubic work is cheap but the Python loop is not
            chosen = "cpu"
        else:
            chosen = "torch"

    if chosen == "triton":
        if not (HAS_TRITON and p4.is_cuda and N <= MAX_BLOCK):
            raise RuntimeError(f"triton backend unavailable (needs triton, CUDA tensors, N <= {MAX_BLOCK})")
        from .triton_backend import cluster_batch_triton

        out = cluster_batch_triton(p4, mask, R=R, p=p)
    elif chosen == "triton-large":
        if not (HAS_TRITON and p4.is_cuda and N <= MAX_LARGE_N):
            raise RuntimeError(f"triton-large backend unavailable (needs triton, CUDA tensors, N <= {MAX_LARGE_N})")
        from .triton_large import cluster_batch_triton_large

        out = cluster_batch_triton_large(p4, mask, R=R, p=p, decode=decode)
    elif chosen == "cpu":
        from .cpu_backend import cluster_batch_cpu

        out = cluster_batch_cpu(p4, mask, R=R, p=p)
    elif chosen == "torch":
        if N > 512:
            import warnings

            warnings.warn(
                f"torch backend is O(N^3) per event; N={N} will be slow and "
                f"allocate (B, N, N) buffers. Use the triton-large backend on GPU.",
                stacklevel=2,
            )
        from .torch_backend import cluster_batch_torch

        out = cluster_batch_torch(p4, mask, R=R, p=p)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    out.setdefault("mask", mask)
    return ClusterOutput(**out)
