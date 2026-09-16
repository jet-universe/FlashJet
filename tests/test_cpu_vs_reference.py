"""Validate the batched NumPy CPU backend.

It runs the same NN-array algorithm as nn_reference/triton_large, so in
float64 it must reproduce nn_reference step for step -- that is a much
stronger check than a partition match, and it is the check that pins the
batched lock-step bookkeeping (compaction, chunking, threading).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flashjet import cluster
from flashjet.cpu_backend import cluster_batch_cpu
from flashjet.nn_reference import cluster_event_nn
from conftest import random_event


def pad_batch(events, dtype=torch.float64):
    N = max((len(e) for e in events), default=1)
    p4 = torch.zeros(len(events), N, 4, dtype=dtype)
    mask = torch.zeros(len(events), N, dtype=torch.bool)
    for b, e in enumerate(events):
        p4[b, : len(e)] = torch.as_tensor(np.asarray(e), dtype=dtype)
        mask[b, : len(e)] = True
    return p4, mask


def partitions(jet_idx, n_jets):
    return {frozenset(np.flatnonzero(jet_idx == j).tolist()) for j in range(n_jets)} - {frozenset()}


@pytest.mark.parametrize("algo,p", [("antikt", -1.0), ("kt", 1.0), ("cambridge", 0.0)])
@pytest.mark.parametrize("R", [0.4, 1.0])
@pytest.mark.parametrize("boosted", [False, True])
def test_cpu_matches_nn_reference_step_for_step(rng, algo, p, R, boosted):
    events = [random_event(rng, int(rng.integers(1, 60)), boosted=boosted) for _ in range(12)]
    p4, mask = pad_batch(events)
    out = cluster_batch_cpu(p4, mask, R=R, p=p)
    for b, ev in enumerate(events):
        n = len(ev)
        ref = cluster_event_nn(ev, R=R, p=p)
        for key in ("hist_p1", "hist_p2", "hist_child", "jet_idx"):
            np.testing.assert_array_equal(out[key][b, :n].numpy(), ref[key], err_msg=key)
        np.testing.assert_allclose(out["hist_d"][b, :n].numpy(), ref["hist_d"], rtol=1e-12)
        assert int(out["n_jets"][b]) == ref["n_jets"]


@pytest.mark.parametrize("R", [0.4, 1.0])
def test_cpu_partitions_match_bruteforce_reference(rng, R):
    events = [random_event(rng, int(rng.integers(1, 50)), boosted=b % 2 == 0) for b in range(10)]
    p4, mask = pad_batch(events)
    out = cluster_batch_cpu(p4, mask, R=R, p=-1.0)
    for b, ev in enumerate(events):
        seq = cluster(ev, R=R, p=-1.0)  # float64 brute force, FastJet-validated
        assert partitions(out["jet_idx"][b].numpy(), int(out["n_jets"][b])) == {
            frozenset(c) for c in seq.jet_constituents()
        }


def test_cpu_padding_and_empty_events(rng):
    """Ragged batch with empty and single-particle events, and trailing pad."""
    events = [np.zeros((0, 4)), random_event(rng, 1), random_event(rng, 30), np.zeros((0, 4))]
    p4, mask = pad_batch(events)
    out = cluster_batch_cpu(p4, mask, R=0.4, p=-1.0)
    assert out["n_jets"].tolist()[0] == 0 and out["n_jets"].tolist()[1] == 1
    assert (out["jet_idx"][~mask] == -1).all()
    for b, ev in enumerate(events):
        n = len(ev)
        if n:
            ref = cluster_event_nn(ev, R=0.4, p=-1.0)
            np.testing.assert_array_equal(out["jet_idx"][b, :n].numpy(), ref["jet_idx"])


@pytest.mark.parametrize("chunk_bytes,threads", [(1 << 22, 1), (1 << 12, 1), (1 << 12, 4)])
def test_cpu_chunking_and_threading_are_transparent(rng, chunk_bytes, threads):
    """Chunk size and thread count are performance knobs only: length-sorted
    chunks are dispatched out of order, so this also pins the row mapping."""
    events = [random_event(rng, int(rng.integers(1, 45))) for _ in range(24)]
    p4, mask = pad_batch(events)
    base = cluster_batch_cpu(p4, mask, R=0.4, p=-1.0)
    got = cluster_batch_cpu(p4, mask, R=0.4, p=-1.0, chunk_bytes=chunk_bytes, threads=threads)
    for key in ("jet_idx", "n_jets", "hist_p1", "hist_p2", "hist_child", "hist_d"):
        assert torch.equal(base[key], got[key]), key


def test_cpu_float32_reproduces_float64_partitions(rng):
    """float32 input halves the memory traffic; near-degenerate merges may
    order differently, but the physics-level partition should survive."""
    events = [random_event(rng, 40) for _ in range(12)]
    p64, mask = pad_batch(events)
    a = cluster_batch_cpu(p64, mask, R=0.4, p=-1.0)
    b = cluster_batch_cpu(p64.float(), mask, R=0.4, p=-1.0)
    assert b["hist_d"].dtype == torch.float32
    mismatches = sum(
        partitions(a["jet_idx"][i].numpy(), int(a["n_jets"][i]))
        != partitions(b["jet_idx"][i].numpy(), int(b["n_jets"][i]))
        for i in range(len(events))
    )
    assert mismatches <= 1


def test_auto_backend_picks_cpu_for_cpu_tensors(rng):
    """The O(N^3) torch backend must not be what CPU users land on."""
    events = [random_event(rng, 40) for _ in range(4)]
    p4, mask = pad_batch(events)
    auto = cluster(p4, mask, R=0.4, algorithm="antikt")
    explicit = cluster(p4, mask, R=0.4, algorithm="antikt", backend="cpu")
    assert torch.equal(auto.jet_idx, explicit.jet_idx)
    assert torch.equal(auto.hist_child, explicit.hist_child)


native = pytest.importorskip("flashjet._native")
requires_native = pytest.mark.skipif(not native.HAS_NATIVE, reason="C++ kernel not built")


@requires_native
@pytest.mark.parametrize("n", [1, 31, 32, 33, 120, 400])
@pytest.mark.parametrize("R", [0.2, 0.4, 1.0])
@pytest.mark.parametrize("boosted", [False, True])
def test_native_matches_numpy_across_the_tiling_threshold(rng, n, R, boosted):
    """The C++ kernel switches to the tiled+heap strategy at n >= 32.  Both
    sides of that switch must reproduce the NumPy mirror's merge history
    exactly -- tiling only ever drops neighbours farther than R, which can
    never win a merge, and the heap breaks ties on (distance, slot) the way
    np.argmin does."""
    ev = random_event(rng, n, boosted=boosted)
    p4, mask = pad_batch([ev])
    got = cluster_batch_cpu(p4, mask, R=R, p=-1.0, native=True)
    ref = cluster_batch_cpu(p4, mask, R=R, p=-1.0, native=False)
    for key in ("hist_p1", "hist_p2", "hist_child", "jet_idx", "n_jets"):
        assert torch.equal(got[key], ref[key]), key
    np.testing.assert_allclose(got["hist_d"], ref["hist_d"], rtol=1e-11)


@requires_native
def test_native_is_thread_safe(rng):
    """Each OpenMP worker owns its scratch and writes to disjoint output rows,
    so thread count is a performance knob and nothing else.  FastJet's own
    ClusterSequence is not safe to share this way, which is the reason to be
    explicit about it here."""
    events = [random_event(rng, int(rng.integers(1, 200))) for _ in range(64)]
    p4, mask = pad_batch(events, dtype=torch.float64)
    serial = cluster_batch_cpu(p4, mask, R=0.4, p=-1.0, threads=1)
    for nt in (2, 4, 8):
        out = cluster_batch_cpu(p4, mask, R=0.4, p=-1.0, threads=nt)
        for key in serial:
            assert torch.equal(serial[key], out[key]), f"threads={nt} changed {key}"

    # and concurrent calls from Python threads must not interfere either
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: cluster_batch_cpu(p4, mask, R=0.4, p=-1.0), range(8)))
    for out in outs:
        assert torch.equal(serial["jet_idx"], out["jet_idx"])


@requires_native
def test_native_handles_beamlike_and_degenerate_input():
    """Zero-pt and collinear particles drive the MaxRap guard and the tile
    grid's degenerate-span path."""
    ev = np.array([
        [0.0, 0.0, 10.0, 10.0],   # exactly beam-like: kt2 == 0
        [0.0, 0.0, -10.0, 10.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],     # exact duplicate -> zero dR
        [1e-8, 1e-8, 0.0, 1.5e-8],
    ])
    p4, mask = pad_batch([ev])
    got = cluster_batch_cpu(p4, mask, R=0.4, p=-1.0, native=True)
    ref = cluster_batch_cpu(p4, mask, R=0.4, p=-1.0, native=False)
    assert torch.equal(got["jet_idx"], ref["jet_idx"])
    assert int(got["n_jets"][0]) >= 1


@requires_native
@pytest.mark.parametrize("n", [50, 200, 600, 2000])
def test_native_float32_agrees_with_the_triton_spec(rng, n):
    """The C++ kernel and triton_large are two implementations of ONE spec:
    nn_reference.  triton_large is float32-only and is pinned to the float32
    mirror by tests/test_triton_large.py; this pins the C++ kernel to the same
    mirror at the same precision, so the CPU and GPU paths cannot drift apart
    without one of the two tests failing.  (The GPU half only runs on a CUDA
    box; this half runs everywhere.)"""
    ev = random_event(rng, n)
    ref = cluster_event_nn(ev, R=0.4, p=-1.0, dtype=np.float32)
    p4, mask = pad_batch([ev], dtype=torch.float32)
    got = cluster_batch_cpu(p4, mask, R=0.4, p=-1.0, native=True)
    assert partitions(got["jet_idx"][0].numpy(), int(got["n_jets"][0])) == partitions(
        ref["jet_idx"], ref["n_jets"]
    )


def test_backends_share_one_output_contract():
    """Every batched backend must return the same keys, and decode jet_idx
    from the merge history with the same decoder -- that is what makes the
    equivalence tests transitive across CPU, C++ and Triton."""
    import inspect

    from flashjet import cpu_backend, torch_backend, triton_large

    keys = {"jet_idx", "n_jets", "hist_p1", "hist_p2", "hist_child", "hist_d"}
    p4, mask = pad_batch([np.zeros((3, 4))])
    assert set(cpu_backend.cluster_batch_cpu(p4, mask, R=0.4, p=-1.0)) == keys
    assert set(torch_backend.cluster_batch_torch(p4, mask, R=0.4, p=-1.0)) == keys
    for mod in (cpu_backend, triton_large):
        assert "jet_idx_from_history" in inspect.getsource(mod), mod.__name__


@requires_native
@pytest.mark.parametrize("R", [0.2, 0.4, 1.0, 1.047, 1.1, 2.0])
def test_native_handles_the_phi_seam(rng, R):
    """The C++ kernel skips the phi-wrap correction for cells whose 3x3 block
    cannot straddle phi = 0, which is most of them.  That shortcut is only
    valid when the block spans less than pi in phi, i.e. when there are enough
    columns -- so R is swept across the threshold where the shortcut turns off
    (2*pi/R < 6, i.e. R > 1.047).  Events are packed against the seam, which
    is where a wrong periodicity flag would show up."""
    n = 300
    pt = rng.uniform(0.5, 80.0, n)
    y = rng.uniform(-1.0, 1.0, n)
    # straddle phi = 0: half just above, half just below
    phi = np.where(rng.random(n) < 0.5, rng.uniform(0, 0.35, n),
                   rng.uniform(2 * np.pi - 0.35, 2 * np.pi, n))
    m = rng.uniform(0.0, 1.0, n)
    mt = np.sqrt(m**2 + pt**2)
    ev = np.stack([pt * np.cos(phi), pt * np.sin(phi),
                   mt * np.sinh(y), mt * np.cosh(y)], axis=1)

    p4, mask = pad_batch([ev])
    got = cluster_batch_cpu(p4, mask, R=R, p=-1.0, native=True)
    ref = cluster_batch_cpu(p4, mask, R=R, p=-1.0, native=False)
    for key in ("hist_p1", "hist_p2", "hist_child", "jet_idx", "n_jets"):
        assert torch.equal(got[key], ref[key]), key
