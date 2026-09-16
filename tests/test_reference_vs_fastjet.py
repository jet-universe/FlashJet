"""Validate the NumPy reference implementation against real FastJet."""

import numpy as np
import pytest

fastjet = pytest.importorskip("fastjet")

from flashjet import cluster
from conftest import random_event

FJ_ALGOS = {
    "antikt": fastjet.antikt_algorithm,
    "kt": fastjet.kt_algorithm,
    "cambridge": fastjet.cambridge_algorithm,
}


def fastjet_partition(p4, R, algo, ptmin=0.0):
    jd = fastjet.JetDefinition(FJ_ALGOS[algo], R)
    pjs = []
    for i, (px, py, pz, E) in enumerate(p4):
        pj = fastjet.PseudoJet(float(px), float(py), float(pz), float(E))
        pj.set_user_index(i)
        pjs.append(pj)
    cs = fastjet.ClusterSequence(pjs, jd)
    jets = fastjet.sorted_by_pt(cs.inclusive_jets(ptmin))
    parts = [frozenset(c.user_index() for c in j.constituents()) for j in jets]
    p4s = np.array([[j.px(), j.py(), j.pz(), j.E()] for j in jets]).reshape(-1, 4)
    return parts, p4s


@pytest.mark.parametrize("algo", ["antikt", "kt", "cambridge"])
@pytest.mark.parametrize("R", [0.4, 1.0])
@pytest.mark.parametrize("boosted", [False, True])
def test_partitions_and_jets_match_fastjet(rng, algo, R, boosted):
    for trial in range(10):
        n = int(rng.integers(1, 64))
        ev = random_event(rng, n, boosted=boosted)

        fj_parts, fj_p4 = fastjet_partition(ev, R, algo)
        seq = cluster(ev, R=R, algorithm=algo)
        my_parts = [frozenset(c) for c in seq.jet_constituents()]
        my_p4 = seq.inclusive_jets()

        assert set(my_parts) == set(fj_parts), f"partition mismatch (n={n}, trial={trial})"
        assert my_p4.shape == fj_p4.shape
        np.testing.assert_allclose(
            np.sort(np.hypot(my_p4[:, 0], my_p4[:, 1])),
            np.sort(np.hypot(fj_p4[:, 0], fj_p4[:, 1])),
            rtol=1e-9,
        )


def test_ptmin_filter(rng):
    ev = random_event(rng, 40)
    seq = cluster(ev, R=0.4, algorithm="antikt")
    jets = seq.inclusive_jets(ptmin=20.0)
    assert all(np.hypot(j[0], j[1]) > 20.0 for j in jets)
