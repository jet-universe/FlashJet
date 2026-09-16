"""Single-event NumPy reference implementation of generalized-kt clustering.

This is a direct, readable transcription of the FastJet N^2 sequential
recombination (see extern/fastjet-*/src/ClusterSequence.cc) used as the
ground truth for the GPU backends.  It is validated against the real FastJet
python bindings in tests/test_reference_vs_fastjet.py.

Distance measure (generalized kt, E-scheme recombination):
    d_ij = min(kt_i^(2p), kt_j^(2p)) * dR_ij^2 / R^2,   dR^2 = dy^2 + dphi^2
    d_iB = kt_i^(2p)
p = -1: anti-kt, p = 0: Cambridge/Aachen, p = 1: kt.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .kinematics import rap_phi_kt2

BEAM = -1


@dataclass
class HistoryStep:
    parent1: int  # pseudojet id
    parent2: int  # pseudojet id, or BEAM (-1) for a beam merge
    child: int    # new pseudojet id, or BEAM for a beam merge
    d: float      # distance at which the merge happened


@dataclass
class ClusterSequenceRef:
    """Result of clustering one event."""

    p4: np.ndarray                 # (n_pseudojets, 4) px, py, pz, E; grows as merges happen
    history: List[HistoryStep] = field(default_factory=list)
    beam_jets: List[int] = field(default_factory=list)  # pseudojet ids merged with the beam, in merge order
    n_initial: int = 0

    def inclusive_jets(self, ptmin: float = 0.0) -> np.ndarray:
        """Jet 4-momenta (pt-sorted, descending) with pt > ptmin."""
        jets = [self.p4[i] for i in self.beam_jets if np.hypot(self.p4[i, 0], self.p4[i, 1]) > ptmin]
        if not jets:
            return np.zeros((0, 4))
        jets = np.array(jets)
        order = np.argsort(-np.hypot(jets[:, 0], jets[:, 1]), kind="stable")
        return jets[order]

    def constituents(self, pseudojet_id: int) -> List[int]:
        """Indices of the initial particles contained in a pseudojet."""
        children = {}
        for h in self.history:
            if h.child != BEAM:
                children[h.child] = (h.parent1, h.parent2)
        stack, out = [pseudojet_id], []
        while stack:
            i = stack.pop()
            if i < self.n_initial:
                out.append(i)
            else:
                stack.extend(children[i])
        return sorted(out)

    def jet_constituents(self, ptmin: float = 0.0) -> List[List[int]]:
        """Constituent index lists, ordered like inclusive_jets(ptmin)."""
        ids = [i for i in self.beam_jets if np.hypot(self.p4[i, 0], self.p4[i, 1]) > ptmin]
        ids.sort(key=lambda i: -np.hypot(self.p4[i, 0], self.p4[i, 1]))
        return [self.constituents(i) for i in ids]


def cluster_event(p4: np.ndarray, R: float = 0.4, p: float = -1.0) -> ClusterSequenceRef:
    """Cluster a single event; p4 is (n, 4) with columns px, py, pz, E."""
    p4 = np.atleast_2d(np.asarray(p4, dtype=np.float64))
    n = len(p4)
    seq = ClusterSequenceRef(p4=p4.copy(), n_initial=n)
    if n == 0:
        return seq

    R2 = R * R
    active = list(range(n))
    mom = [p4[i] for i in range(n)]

    while active:
        a = np.array([mom[i] for i in active])
        rap, phi, kt2 = rap_phi_kt2(a[:, 0], a[:, 1], a[:, 2], a[:, 3])
        w = np.maximum(kt2, 1e-30) ** p  # kt^(2p), floored like every other rung

        drap = rap[:, None] - rap[None, :]
        dphi = np.abs(phi[:, None] - phi[None, :])
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        dij = np.minimum(w[:, None], w[None, :]) * (drap**2 + dphi**2) / R2
        np.fill_diagonal(dij, np.inf)

        # per-slot candidate with the beam merge as the default: a pair
        # displaces it only on strictly smaller d (FastJet keeps NN = NULL
        # at dist == R^2, ClusterSequence.hh), slot ties break low
        d_pair = dij.min(axis=1)
        cand = np.minimum(d_pair, w)
        s = int(np.argmin(cand))
        dmin = float(cand[s])

        if w[s] <= d_pair[s]:  # beam merge
            i = active[s]
            seq.history.append(HistoryStep(i, BEAM, BEAM, dmin))
            seq.beam_jets.append(i)
            active.remove(i)
        else:  # pair merge (E-scheme: 4-momentum sum)
            i, j = active[s], active[int(np.argmin(dij[s]))]
            child = len(mom)
            combined = mom[i] + mom[j]
            mom.append(combined)
            seq.p4 = np.vstack([seq.p4, combined])
            seq.history.append(HistoryStep(i, j, child, dmin))
            active.remove(i)
            active.remove(j)
            active.append(child)

    return seq
