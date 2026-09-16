"""The compiled decode (FLASHJET_COMPILE_DECODE=1) must be a pure speedup:
bitwise-identical to eager, and — the CUDA-graph footgun — outputs must
survive later calls (graph replay overwrites the graph-owned buffers, so
the dispatcher clones; without that, holding jet_idx across batches would
silently corrupt it)."""

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

pytest.importorskip("triton")

import flashjet.history as fhist
from flashjet.triton_large import cluster_batch_triton_large
from test_torch_vs_reference import make_batch


def test_compiled_decode_bitwise_and_replay_safe(rng, monkeypatch):
    _, p4a, ma = make_batch(rng, B=8, Nmax=64)
    _, p4b, mb = make_batch(rng, B=8, Nmax=64)
    p4a, ma = p4a.float().cuda(), ma.cuda()
    p4b, mb = p4b.float().cuda(), mb.cuda()

    monkeypatch.delenv("FLASHJET_COMPILE_DECODE", raising=False)
    ea = cluster_batch_triton_large(p4a, ma, R=0.4, p=-1.0)
    eb = cluster_batch_triton_large(p4b, mb, R=0.4, p=-1.0)

    monkeypatch.setenv("FLASHJET_COMPILE_DECODE", "1")
    monkeypatch.setattr(fhist, "_compiled", None)
    ca = cluster_batch_triton_large(p4a, ma, R=0.4, p=-1.0)
    cb = cluster_batch_triton_large(p4b, mb, R=0.4, p=-1.0)  # replays the graph

    # ca must NOT have been overwritten by cb's replay, and both match eager
    for got, ref in ((ca, ea), (cb, eb)):
        for k in ("jet_idx", "n_jets"):
            assert torch.equal(got[k], ref[k])
