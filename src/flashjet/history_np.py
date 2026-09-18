"""NumPy decode of the merge history: particle -> jet, without torch.

``history.jet_idx_from_history`` does this for batched torch tensors.  The same
walk is useful where torch is not available -- a NumPy-only install, or a
serving process that runs the C++ kernel through :mod:`flashjet._native` -- so
it is repeated here for one event at a time, in plain NumPy.

The history is the one every backend produces: ``hist_p1``/``hist_p2`` are the
parents of step ``k`` (pseudojet ids; the initial particles are ``0..n-1`` and
each pair merge creates the next id), ``hist_p2 == -1`` marks a beam merge, and
``hist_child`` is the id a pair merge created.
"""

import numpy as np


def jet_idx_from_history_np(hist_p1, hist_p2, hist_child, n):
    """Particle -> jet index for one event, in beam-merge order.

    Args:
        hist_p1, hist_p2, hist_child: integer arrays of at least ``n`` steps.
        n: number of initial particles.
    Returns:
        ``(jet_idx, n_jets)``: ``jet_idx`` is an (n,) int32 array, ``n_jets``
        the number of jets.

    The history is walked backwards: a beam merge names the jet of its
    pseudojet, and a pair merge hands its child's jet down to both parents, so
    one pass is enough.
    """
    hist_p1 = np.asarray(hist_p1)
    hist_p2 = np.asarray(hist_p2)
    hist_child = np.asarray(hist_child)
    if n <= 0:
        return np.zeros(0, dtype=np.int32), 0

    beam = hist_p2[:n] < 0
    n_jets = int(beam.sum())
    # ids run 0..2n-2: n initial particles plus one per pair merge
    jet_of = np.full(2 * n, -1, dtype=np.int32)
    jet = n_jets
    for step in range(n - 1, -1, -1):
        if beam[step]:
            jet -= 1
            jet_of[hist_p1[step]] = jet
        else:
            owner = jet_of[hist_child[step]]
            jet_of[hist_p1[step]] = owner
            jet_of[hist_p2[step]] = owner
    return jet_of[:n].copy(), n_jets
