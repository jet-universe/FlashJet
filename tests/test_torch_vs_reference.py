"""Validate the batched torch backend against the NumPy reference."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flashjet import cluster
from flashjet.torch_backend import cluster_batch_torch
from conftest import random_event


def make_batch(rng, B, Nmax, boosted=False):
    events = [random_event(rng, int(rng.integers(1, Nmax + 1)), boosted=boosted) for _ in range(B)]
    p4 = torch.zeros(B, Nmax, 4, dtype=torch.float64)
    mask = torch.zeros(B, Nmax, dtype=torch.bool)
    for b, ev in enumerate(events):
        p4[b, : len(ev)] = torch.from_numpy(ev)
        mask[b, : len(ev)] = True
    return events, p4, mask


def torch_partitions(out, mask, b):
    n = int(mask[b].sum())
    parts = []
    for j in range(int(out["n_jets"][b])):
        members = frozenset(i for i in range(n) if int(out["jet_idx"][b, i]) == j)
        if members:
            parts.append(members)
    return parts


@pytest.mark.parametrize("algo,p", [("antikt", -1.0), ("kt", 1.0), ("cambridge", 0.0)])
@pytest.mark.parametrize("R", [0.4, 1.0])
def test_torch_matches_reference(rng, algo, p, R):
    events, p4, mask = make_batch(rng, B=16, Nmax=48)
    out = cluster_batch_torch(p4, mask, R=R, p=p)

    for b, ev in enumerate(events):
        seq = cluster(ev, R=R, p=p)
        ref_parts = {frozenset(c) for c in seq.jet_constituents()}
        got_parts = set(torch_partitions(out, mask, b))
        assert got_parts == ref_parts, f"event {b}: partition mismatch"
        assert int(out["n_jets"][b]) == len(seq.beam_jets)


def test_jets_p4_scatter_add_and_grad(rng):
    events, p4, mask = make_batch(rng, B=8, Nmax=32)
    p4 = p4.clone().requires_grad_(True)
    res = cluster(p4, mask, R=0.4, algorithm="antikt", backend="torch")
    jets = res.jets_p4(p4)

    # E-scheme jets are exactly the sum of their constituents
    for b, ev in enumerate(events):
        seq = cluster(ev, R=0.4, algorithm="antikt")
        ref = np.sort(np.hypot(*np.asarray(seq.inclusive_jets())[:, :2].T)) if seq.beam_jets else np.zeros(0)
        got = torch.hypot(jets[b, :, 0], jets[b, :, 1])
        got = np.sort(got[got > 1e-9].detach().numpy())
        np.testing.assert_allclose(got, ref[ref > 1e-9], rtol=1e-9)

    # gradients flow through jet momenta to the inputs
    jets.sum().backward()
    assert p4.grad is not None
    assert torch.isfinite(p4.grad).all()
    assert p4.grad[mask].abs().sum() > 0


def test_history_consistency(rng):
    _, p4, mask = make_batch(rng, B=8, Nmax=32)
    out = cluster_batch_torch(p4, mask, R=0.4, p=-1.0)
    n = mask.sum(1)
    for b in range(p4.shape[0]):
        nb = int(n[b])
        # exactly n merge steps, the rest padded
        assert (out["hist_p1"][b, :nb] >= 0).all()
        assert (out["hist_p1"][b, nb:] == -2).all()
        # number of beam merges == number of jets
        assert int((out["hist_p2"][b, :nb] == -1).sum()) == int(out["n_jets"][b])
        # d_min is non-decreasing? (not guaranteed in general; just finite)
        assert torch.isfinite(out["hist_d"][b, :nb]).all()


def test_sort_jets_by_pt(rng):
    _, p4, mask = make_batch(rng, B=4, Nmax=24)
    res = cluster(p4, mask, R=0.4, backend="torch")
    jets = res.jets_p4(p4)
    sjets, order = res.sort_jets_by_pt(jets)
    pt = torch.hypot(sjets[..., 0], sjets[..., 1])
    assert (pt[:, :-1] >= pt[:, 1:] - 1e-12).all()
