"""Nearest-neighbour (FastJet "N^2 strategy") clustering — NumPy mirror.

This implements the exact algorithm used by the large-N Triton kernel
(triton_large.py) so its logic can be validated on CPU against the brute
force reference:

Like FastJet's N2Plain strategy, neighbours are maintained in the GEOMETRIC
metric dR^2, not in the d_ij measure:

  * Cacciari-Salam lemma: the globally minimal d_ij is always realized by
    some slot together with its geometric nearest neighbour (if the winning
    pair (a, b) with w_a <= w_b were not geometric neighbours, a closer c
    would give the strictly smaller d_ac).  So the per-slot candidate
    min(w_i, w_gnn(i)) * dR_gnn^2 / R^2  (vs the beam distance w_i) is a
    valid pair distance and its minimum over slots is the global minimum.
  * geometric NNs only go stale when a slot's POSITION changes or dies; a
    merge changing only momenta invalidates nothing.  Average stale count
    per step is O(1), giving O(N^2) total work.  (Maintaining NNs in the
    d_ij measure instead is also correct but cascades: every soft particle
    points at the hard core, so each core merge forces O(N) rescans.)

State lives in fixed slot arrays (a merged pseudojet overwrites slot i, slot
j dies) exactly like the kernels do.
"""

import numpy as np

from .kinematics import rap_phi_kt2

BEAM = -1
PAD = -2
TWO_PI = 2 * np.pi


def _pairwise_dr2(rap_i, phi_i, rap, phi, ok):
    dphi = np.abs(phi_i - phi)
    dphi = np.minimum(dphi, TWO_PI - dphi)
    return np.where(ok, (rap_i - rap) ** 2 + dphi**2, np.inf)


def cluster_event_nn(p4, R=0.4, p=-1.0, dtype=np.float64):
    """Cluster one event; returns the same field names as the batched backends
    (1-D arrays instead of (B, N))."""
    p4 = np.atleast_2d(np.asarray(p4, dtype=dtype)).copy()
    n = len(p4)
    inv_R2 = dtype(1.0 / (R * R))

    jet_idx = np.full(n, BEAM, dtype=np.int64)
    n_jets = 0
    hist_p1 = np.full(n, PAD, dtype=np.int64)
    hist_p2 = np.full(n, PAD, dtype=np.int64)
    hist_child = np.full(n, PAD, dtype=np.int64)
    hist_d = np.zeros(n, dtype=dtype)
    if n == 0:
        return dict(jet_idx=jet_idx, n_jets=0, hist_p1=hist_p1, hist_p2=hist_p2,
                    hist_child=hist_child, hist_d=hist_d)

    px, py, pz, E = (p4[:, k] for k in range(4))
    rap, phi, kt2 = rap_phi_kt2(px, py, pz, E)
    rap, phi = rap.astype(dtype), phi.astype(dtype)
    w = np.maximum(kt2, 1e-30).astype(dtype) ** dtype(p)  # kt^(2p) == d_iB

    active = np.ones(n, dtype=bool)
    ids = np.arange(n, dtype=np.int64)
    next_id = n
    dest = np.arange(n, dtype=np.int64)
    idx = np.arange(n)

    def rescan(k):
        ok = active & (idx != k)
        dr2 = _pairwise_dr2(rap[k], phi[k], rap, phi, ok)
        j = int(np.argmin(dr2))
        gnn_dr2[k] = dr2[j]
        gnn_idx[k] = j if np.isfinite(dr2[j]) else -1

    gnn_dr2 = np.full(n, np.inf, dtype=dtype)
    gnn_idx = np.full(n, -1, dtype=np.int64)
    for k in range(n):
        rescan(k)

    for step in range(n):
        # candidate pair distance per slot via its geometric NN
        w_gnn = np.where(gnn_idx >= 0, w[np.maximum(gnn_idx, 0)], np.inf)
        d_pair = np.minimum(w, w_gnn) * gnn_dr2 * inv_R2
        cand = np.where(active, np.minimum(d_pair, w), np.inf)
        i = int(np.argmin(cand))
        gmin = cand[i]
        is_pair = d_pair[i] < w[i]

        if is_pair:
            j = int(gnn_idx[i])
            hist_p1[step], hist_p2[step], hist_child[step], hist_d[step] = ids[i], ids[j], next_id, gmin

            p4[i] += p4[j]
            active[j] = False
            r, f, k2 = rap_phi_kt2(p4[i, 0], p4[i, 1], p4[i, 2], p4[i, 3])
            rap[i], phi[i] = dtype(r), dtype(f)
            w[i] = np.maximum(k2, 1e-30) ** dtype(p)
            ids[i] = next_id
            next_id += 1
            dest[dest == j] = i

            ok = active & (idx != i)
            stale = ok & ((gnn_idx == i) | (gnn_idx == j))
            dr2_new = _pairwise_dr2(rap[i], phi[i], rap, phi, ok)
            improve = ok & ~stale & (dr2_new < gnn_dr2)
            gnn_dr2[improve] = dr2_new[improve]
            gnn_idx[improve] = i
            # geometric NN of the new pseudojet falls out of the same row
            jbest = int(np.argmin(dr2_new))
            gnn_dr2[i] = dr2_new[jbest]
            gnn_idx[i] = jbest if np.isfinite(dr2_new[jbest]) else -1
            for k in np.flatnonzero(stale):
                rescan(k)
        else:
            hist_p1[step], hist_p2[step], hist_child[step], hist_d[step] = ids[i], BEAM, BEAM, gmin
            active[i] = False
            jet_idx[dest == i] = n_jets
            n_jets += 1
            stale = active & (gnn_idx == i)
            for k in np.flatnonzero(stale):
                rescan(k)

    return dict(jet_idx=jet_idx, n_jets=n_jets, hist_p1=hist_p1, hist_p2=hist_p2,
                hist_child=hist_child, hist_d=hist_d)
