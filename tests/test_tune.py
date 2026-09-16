"""The autotune path must stay opt-in and reproducible: the winner comes
only from the curated (validated) config list, is persisted per (GPU model,
N-band), and later calls read it back so outputs are bitwise-stable —
per-process re-tuning would silently flip jet_idx on near-tie merges."""

import json
import os

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

pytest.importorskip("triton")

import flashjet.tune as ftune
from flashjet.triton_large import cluster_batch_triton_large
from test_torch_vs_reference import make_batch


@pytest.fixture
def tune_cache(tmp_path, monkeypatch):
    path = tmp_path / "tune.json"
    monkeypatch.setenv("FLASHJET_TUNE_CACHE", str(path))
    monkeypatch.delenv("FLASHJET_TUNE", raising=False)
    monkeypatch.setattr(ftune, "SPIN_SECONDS", 0.2)
    monkeypatch.setattr(ftune, "_proc_cache", {})
    return path


def test_tune_persists_and_reproduces(rng, tune_cache):
    _, p4, mask = make_batch(rng, B=32, Nmax=200)
    p4 = p4.float().cuda()
    mask = mask.cuda()
    a = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0, tune=True)

    (key, entry), = json.load(open(tune_cache)).items()
    assert "band1" in key
    cfg = (entry["block"], entry["num_warps"], tuple(entry["tile"]))
    assert cfg in [(b, w, t) for b, w, t in ftune.CONFIGS]

    # second call hits the persisted cache (fresh process simulated by
    # clearing the in-process memo) and must reproduce bitwise
    ftune._proc_cache.clear()
    b = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0, tune=True)
    for k in a:
        assert torch.equal(a[k], b[k])


def test_tune_off_by_default(rng, tune_cache):
    _, p4, mask = make_batch(rng, B=8, Nmax=64)
    p4 = p4.float().cuda()
    mask = mask.cuda()
    a = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0)
    assert not os.path.exists(tune_cache)
    # and the default path is exactly the banded default, bitwise
    b = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0, block=1024, num_warps=2, tile=(32, 64))
    for k in a:
        assert torch.equal(a[k], b[k])


def test_explicit_knobs_skip_tuning(rng, tune_cache):
    _, p4, mask = make_batch(rng, B=8, Nmax=64)
    p4 = p4.float().cuda()
    mask = mask.cuda()
    cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0, tune=True, block=512)
    assert not os.path.exists(tune_cache)
