"""Validate the NN-array (large-N) algorithm mirror against the brute-force
reference. This is the CPU proof of the algorithm used by triton_large."""

import numpy as np
import pytest

from flashjet import cluster
from flashjet.nn_reference import cluster_event_nn
from conftest import random_event


def partitions_from_jet_idx(jet_idx, n_jets):
    return {frozenset(np.flatnonzero(jet_idx == j).tolist()) for j in range(n_jets)} - {frozenset()}


@pytest.mark.parametrize("algo,p", [("antikt", -1.0), ("kt", 1.0), ("cambridge", 0.0)])
@pytest.mark.parametrize("R", [0.4, 1.0])
@pytest.mark.parametrize("boosted", [False, True])
def test_nn_matches_reference(rng, algo, p, R, boosted):
    for _ in range(8):
        n = int(rng.integers(1, 80))
        ev = random_event(rng, n, boosted=boosted)
        seq = cluster(ev, R=R, p=p)
        ref_parts = {frozenset(c) for c in seq.jet_constituents()}
        out = cluster_event_nn(ev, R=R, p=p)
        got_parts = partitions_from_jet_idx(out["jet_idx"], out["n_jets"])
        assert got_parts == ref_parts
        assert out["n_jets"] == len(seq.beam_jets)
        # history: same multiset of merge distances
        np.testing.assert_allclose(
            np.sort(out["hist_d"][:n]), np.sort([h.d for h in seq.history]), rtol=1e-9
        )


def test_nn_medium_event(rng):
    ev = random_event(rng, 400)
    seq = cluster(ev, R=0.4, algorithm="antikt")
    out = cluster_event_nn(ev, R=0.4, p=-1.0)
    assert partitions_from_jet_idx(out["jet_idx"], out["n_jets"]) == {
        frozenset(c) for c in seq.jet_constituents()
    }


def test_nn_float32_consistency(rng):
    # float32 (kernel precision) should still reproduce the f64 partitions on
    # generic events
    mismatches = 0
    for _ in range(10):
        ev = random_event(rng, 60)
        a = cluster_event_nn(ev, R=0.4, p=-1.0, dtype=np.float64)
        b = cluster_event_nn(ev, R=0.4, p=-1.0, dtype=np.float32)
        if partitions_from_jet_idx(a["jet_idx"], a["n_jets"]) != partitions_from_jet_idx(
            b["jet_idx"], b["n_jets"]
        ):
            mismatches += 1
    assert mismatches <= 1
