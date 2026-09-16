"""The history decoder must reproduce the kernel-maintained jet assignment.

Validated on CPU against the torch backend, whose jet_idx and history are
themselves validated against the brute-force reference.
"""

import pytest

torch = pytest.importorskip("torch")

from flashjet.history import jet_idx_from_history
from flashjet.torch_backend import cluster_batch_torch
from test_torch_vs_reference import make_batch


@pytest.mark.parametrize("p", [-1.0, 0.0, 1.0])
def test_decode_matches_torch_backend(rng, p):
    _, p4, mask = make_batch(rng, B=24, Nmax=64)
    out = cluster_batch_torch(p4, mask, R=0.4, p=p)
    jet_idx, n_jets = jet_idx_from_history(
        out["hist_p1"], out["hist_p2"], out["hist_child"], mask
    )
    assert torch.equal(jet_idx, out["jet_idx"])
    assert torch.equal(n_jets, out["n_jets"])


def test_decode_edge_cases(rng):
    # single-particle events and full-width events
    _, p4, mask = make_batch(rng, B=8, Nmax=4)
    mask[0, 1:] = False  # one particle
    p4[0, 1:] = 0
    out = cluster_batch_torch(p4, mask, R=0.4, p=-1.0)
    jet_idx, n_jets = jet_idx_from_history(
        out["hist_p1"], out["hist_p2"], out["hist_child"], mask
    )
    assert torch.equal(jet_idx, out["jet_idx"])
    assert torch.equal(n_jets, out["n_jets"])
