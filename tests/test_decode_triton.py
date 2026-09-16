"""The triton decode is a pure launch-count optimization; any deviation from
the eager decode is a bug, not a tolerance.

The decode is integer-only (merge-tree pointer jumping), so every comparison
here is torch.equal on the SAME history tensors: the eager torch-op loop
(_decode, itself pinned to the torch backend in test_history_decode.py) is
the spec, the single-launch Triton kernel must reproduce it bitwise.
"""

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

pytest.importorskip("triton")

from flashjet.history import _decode, _decode_triton, jet_idx_from_history
from flashjet.torch_backend import cluster_batch_torch
from flashjet.triton_large import cluster_batch_triton_large
from test_torch_vs_reference import make_batch
from conftest import random_event


def assert_bitwise(hist, mask):
    hp1, hp2, hch = hist["hist_p1"].cuda(), hist["hist_p2"].cuda(), hist["hist_child"].cuda()
    mask = mask.cuda()
    ref_idx, ref_nj = _decode(hp1, hp2, hch, mask)
    got_idx, got_nj = _decode_triton(hp1, hp2, hch, mask)
    assert torch.equal(got_idx, ref_idx)
    assert torch.equal(got_nj, ref_nj)


@pytest.mark.parametrize("p", [-1.0, 0.0, 1.0])
def test_decode_torch_backend_history(rng, p):
    _, p4, mask = make_batch(rng, B=24, Nmax=64)
    out = cluster_batch_torch(p4, mask, R=0.4, p=p)
    assert_bitwise(out, mask)


@pytest.mark.parametrize("n", [600, 6000])
def test_decode_large_kernel_history(rng, n):
    ev = random_event(rng, n)
    p4 = torch.zeros(2, n, 4)
    p4[0] = p4[1] = torch.from_numpy(ev).float()
    mask = torch.ones(2, n, dtype=torch.bool)
    out = cluster_batch_triton_large(p4.cuda(), mask.cuda(), R=0.4, p=-1.0)
    assert_bitwise(out, mask)


def test_decode_single_particle_events(rng):
    _, p4, mask = make_batch(rng, B=8, Nmax=4)
    mask[0, 1:] = False  # one particle
    p4[0, 1:] = 0
    out = cluster_batch_torch(p4, mask, R=0.4, p=-1.0)
    assert_bitwise(out, mask)


def test_decode_empty_events_in_batch(rng):
    _, p4, mask = make_batch(rng, B=8, Nmax=16)
    mask[2] = False  # n=0 rows mixed into the batch
    p4[2] = 0
    mask[5] = False
    p4[5] = 0
    out = cluster_batch_torch(p4, mask, R=0.4, p=-1.0)
    assert_bitwise(out, mask)


def test_decode_deterministic(rng):
    _, p4, mask = make_batch(rng, B=24, Nmax=64)
    out = cluster_batch_torch(p4, mask, R=0.4, p=-1.0)
    hp1, hp2, hch = out["hist_p1"].cuda(), out["hist_p2"].cuda(), out["hist_child"].cuda()
    mask = mask.cuda()
    first_idx, first_nj = _decode_triton(hp1, hp2, hch, mask)
    for _ in range(5):
        idx, nj = _decode_triton(hp1, hp2, hch, mask)
        assert torch.equal(idx, first_idx)
        assert torch.equal(nj, first_nj)


def test_dispatcher_routes_cuda_to_triton(rng, monkeypatch):
    """jet_idx_from_history on CUDA must give the same (bitwise) answer as
    eager -- i.e. routing through the kernel changes nothing observable."""
    monkeypatch.delenv("FLASHJET_COMPILE_DECODE", raising=False)
    _, p4, mask = make_batch(rng, B=24, Nmax=64)
    out = cluster_batch_torch(p4, mask, R=0.4, p=-1.0)
    hp1, hp2, hch = out["hist_p1"].cuda(), out["hist_p2"].cuda(), out["hist_child"].cuda()
    mask = mask.cuda()
    idx, nj = jet_idx_from_history(hp1, hp2, hch, mask)
    ref_idx, ref_nj = _decode(hp1, hp2, hch, mask)
    assert torch.equal(idx, ref_idx)
    assert torch.equal(nj, ref_nj)
