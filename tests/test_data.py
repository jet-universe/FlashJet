"""Collation and GPU-feeding pipeline tests (collation runs on CPU)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flashjet.data import collate, to_gpu_batches, gpu_batch_ready
from conftest import random_event


def naive_collate(events, n_max):
    B = len(events)
    p4 = torch.zeros(B, n_max, 4)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for b, ev in enumerate(events):
        n = min(len(ev), n_max)
        p4[b, :n] = torch.from_numpy(np.asarray(ev[:n], dtype=np.float32))
        mask[b, :n] = True
    return p4, mask


def test_collate_list_matches_naive(rng):
    events = [random_event(rng, int(rng.integers(1, 50))) for _ in range(32)]
    p4, mask = collate(events)
    ref_p4, ref_mask = naive_collate(events, max(len(e) for e in events))
    assert torch.equal(mask, ref_mask)
    assert torch.equal(p4, ref_p4)


def test_collate_awkward(rng):
    ak = pytest.importorskip("awkward")
    events = [random_event(rng, int(rng.integers(1, 30))) for _ in range(16)]
    arr = ak.Array(
        [[{"px": r[0], "py": r[1], "pz": r[2], "E": r[3]} for r in ev] for ev in events]
    )
    p4, mask = collate(arr)
    ref_p4, ref_mask = naive_collate(events, max(len(e) for e in events))
    assert torch.equal(mask, ref_mask)
    torch.testing.assert_close(p4, ref_p4)


def test_collate_truncate_pt(rng):
    events = [random_event(rng, 40)]
    p4, mask = collate(events, n_max=10, truncate="pt")
    assert mask[0].sum() == 10
    kept_pt = torch.hypot(p4[0, :10, 0], p4[0, :10, 1]).numpy()
    ev32 = events[0].astype(np.float32)
    all_pt = np.hypot(ev32[:, 0], ev32[:, 1])
    np.testing.assert_allclose(np.sort(kept_pt), np.sort(all_pt)[-10:], rtol=1e-6)


def test_collate_out_buffer(rng):
    events = [random_event(rng, int(rng.integers(1, 20))) for _ in range(8)]
    buf = (torch.full((16, 32, 4), 7.0), torch.ones(16, 32, dtype=torch.bool))
    p4, mask = collate(events, n_max=32, out=buf)
    assert p4.shape == (8, 32, 4) and mask.shape == (8, 32)
    ref_p4, ref_mask = naive_collate(events, 32)
    assert torch.equal(p4, ref_p4) and torch.equal(mask, ref_mask)


def test_to_gpu_batches_cpu_fallback(rng):
    events = [random_event(rng, int(rng.integers(1, 20))) for _ in range(10)]
    seen = 0
    for batch in to_gpu_batches(events, batch_size=4, device="cpu"):
        p4, mask = batch[0], batch[1]
        seen += p4.shape[0]
        assert p4.shape[0] in (4, 2)
    assert seen == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_to_gpu_batches_matches_sync(rng):
    import flashjet

    events = [random_event(rng, int(rng.integers(5, 60))) for _ in range(64)]
    outs = []
    for batch in to_gpu_batches(events, batch_size=16):
        p4, mask = gpu_batch_ready(batch)
        out = flashjet.cluster(p4, mask, R=0.4, algorithm="antikt")
        outs.append(out.n_jets.cpu())
    got = torch.cat(outs)

    ref = []
    for s in range(0, 64, 16):
        p4, mask = collate(events[s : s + 16], n_max=None)
        out = flashjet.cluster(p4.cuda(), mask.cuda(), R=0.4, algorithm="antikt")
        ref.append(out.n_jets.cpu())
    assert torch.equal(got, torch.cat(ref))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_scatter_matches_cpu_collation(rng):
    """The GPU-side collation (_scatter_gpu) is bitwise-identical to the CPU
    numpy scatter, across many batches (each ring slot reused several times)
    and deterministic across repeat passes -- the async-double-buffer bug class
    that small/single-batch tests miss."""
    events = [random_event(rng, int(rng.integers(1, 60))) for _ in range(200)]
    bs = 16  # ~13 batches -> each of the 2 ring slots reused ~6x

    def gpu_pass():
        out = []
        for b in to_gpu_batches(events, batch_size=bs, device="cuda"):
            p4, mask = gpu_batch_ready(b)
            out.append((p4.cpu().clone(), mask.cpu().clone()))
        return out

    cpu = [(b[0].clone(), b[1].clone()) for b in to_gpu_batches(events, batch_size=bs, device="cpu")]
    g1, g2 = gpu_pass(), gpu_pass()
    assert len(cpu) == len(g1) == len(g2) > 5
    for (cp, cm), (p1, m1), (p2, m2) in zip(cpu, g1, g2):
        assert torch.equal(cp, p1) and torch.equal(cm, m1)   # GPU == CPU, bitwise
        assert torch.equal(p1, p2) and torch.equal(m1, m2)   # deterministic
