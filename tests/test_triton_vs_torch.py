"""Validate the fused Triton kernel against the torch backend (GPU only)."""

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required for the triton backend", allow_module_level=True)

triton = pytest.importorskip("triton")

from flashjet.torch_backend import cluster_batch_torch
from flashjet.triton_backend import cluster_batch_triton
from test_torch_vs_reference import make_batch


@pytest.mark.parametrize("p", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("R", [0.4, 1.0])
def test_triton_matches_torch(rng, p, R):
    _, p4, mask = make_batch(rng, B=256, Nmax=96)
    p4 = p4.float().cuda()
    mask = mask.cuda()

    got = cluster_batch_triton(p4, mask, R=R, p=p)
    ref = cluster_batch_torch(p4, mask, R=R, p=p)

    # float32 reduction-order differences can flip near-degenerate merges in
    # rare events; require near-perfect agreement and report outliers.
    same = (got["jet_idx"] == ref["jet_idx"]).all(dim=1) & (got["n_jets"] == ref["n_jets"])
    frac = same.float().mean().item()
    assert frac >= 0.98, f"only {frac:.3f} of events match exactly"


def test_triton_deterministic(rng):
    _, p4, mask = make_batch(rng, B=64, Nmax=64)
    p4 = p4.float().cuda()
    mask = mask.cuda()
    a = cluster_batch_triton(p4, mask, R=0.4, p=-1.0)
    b = cluster_batch_triton(p4, mask, R=0.4, p=-1.0)
    for k in a:
        assert torch.equal(a[k], b[k])


def test_triton_history_matches(rng):
    _, p4, mask = make_batch(rng, B=32, Nmax=48)
    p4 = p4.float().cuda()
    mask = mask.cuda()
    got = cluster_batch_triton(p4, mask, R=0.4, p=-1.0)
    ref = cluster_batch_torch(p4, mask, R=0.4, p=-1.0)
    same = (got["jet_idx"] == ref["jet_idx"]).all(dim=1)
    for k in ("hist_p1", "hist_p2", "hist_child"):
        assert (got[k][same] == ref[k][same]).float().mean() > 0.999
