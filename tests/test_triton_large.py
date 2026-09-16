"""Validate the large-N scratch-memory Triton kernel (GPU only)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

pytest.importorskip("triton")

from flashjet.nn_reference import cluster_event_nn
from flashjet.torch_backend import cluster_batch_torch
from flashjet.triton_large import cluster_batch_triton_large
from test_torch_vs_reference import make_batch
from test_nn_vs_reference import partitions_from_jet_idx
from conftest import random_event


@pytest.mark.parametrize("p", [-1.0, 0.0, 1.0])
def test_large_matches_torch_small_n(rng, p):
    _, p4, mask = make_batch(rng, B=128, Nmax=96)
    p4 = p4.float().cuda()
    mask = mask.cuda()
    got = cluster_batch_triton_large(p4, mask, R=0.4, p=p)
    ref = cluster_batch_torch(p4, mask, R=0.4, p=p)
    same = (got["jet_idx"] == ref["jet_idx"]).all(dim=1) & (got["n_jets"] == ref["n_jets"])
    assert same.float().mean().item() >= 0.98


def test_large_pad_contract_all_keys(rng):
    """All six output keys vs the torch backend, including the padding tail
    the in-kernel history prefill must honor: hist_p1/p2/child == PAD (-2)
    and hist_d == 0.0 for every step >= n_init, with one all-False mask row
    (n_init = 0) whose history must come out entirely PAD.  In-range hist_d
    (and merge order on near-ties) legitimately differs from torch in the
    last ulp -- the kernel's w/inv_R2 formulation is its own float32
    transcription -- so the in-range bar stays the usual match fraction."""
    _, p4, mask = make_batch(rng, B=128, Nmax=96)
    mask[7] = False  # empty event: the kernel writes no history steps at all
    p4 = p4.float().cuda()
    mask = mask.cuda()
    got = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0)
    ref = cluster_batch_torch(p4, mask, R=0.4, p=-1.0)

    same = (got["jet_idx"] == ref["jet_idx"]).all(dim=1) & (got["n_jets"] == ref["n_jets"])
    assert same.float().mean().item() >= 0.98

    # the padding tail is contract, not floating point: exact in every event
    pad = torch.arange(mask.shape[1], device=mask.device).expand_as(mask) >= mask.sum(1, keepdim=True)
    for k in got:
        assert got[k].dtype == ref[k].dtype, k
    for k in ("hist_p1", "hist_p2", "hist_child"):
        assert (got[k][pad] == -2).all(), k
        assert (ref[k][pad] == -2).all(), k
    assert (got["hist_d"][pad] == 0.0).all()

    # empty event: no merges happen, so no float arithmetic is involved and
    # every key must equal the torch backend's exactly
    for k in got:
        assert torch.equal(got[k][7], ref[k][7]), k


@pytest.mark.parametrize("n", [600, 2000, 6000])
def test_large_matches_nn_reference(rng, n):
    """Single big events vs the CPU-validated NN mirror (float32 both sides)."""
    ev = random_event(rng, n)
    p4 = torch.zeros(1, n, 4)
    p4[0] = torch.from_numpy(ev).float()
    mask = torch.ones(1, n, dtype=torch.bool)
    got = cluster_batch_triton_large(p4.cuda(), mask.cuda(), R=0.4, p=-1.0)
    ref = cluster_event_nn(ev, R=0.4, p=-1.0, dtype=np.float32)

    gp = partitions_from_jet_idx(got["jet_idx"][0].cpu().numpy(), int(got["n_jets"][0]))
    rp = partitions_from_jet_idx(ref["jet_idx"], ref["n_jets"])
    # float32 chunked reductions can flip near-degenerate merges; demand that
    # the overwhelming majority of jets are identical
    inter = len(gp & rp)
    assert inter / max(len(rp), 1) >= 0.97, f"{inter}/{len(rp)} jets match"
    assert abs(int(got["n_jets"][0]) - ref["n_jets"]) <= max(2, 0.01 * ref["n_jets"])


def test_large_matches_nn_reference_boosted(rng):
    """Boosted (collimated) sprays maximize per-step stale-NN counts -- every
    merge of the dense core invalidates neighbours -- which is exactly the
    surface the batched stale-row rescan restructured.  Boosted events hold
    only ~15 jets each, so the standard match fraction is asserted over the
    aggregate of 8 events, plus exact determinism."""
    B, n = 8, 2000
    evs = [random_event(rng, n, boosted=True) for _ in range(B)]
    p4 = torch.zeros(B, n, 4)
    for i, ev in enumerate(evs):
        p4[i] = torch.from_numpy(ev).float()
    mask = torch.ones(B, n, dtype=torch.bool)
    got = cluster_batch_triton_large(p4.cuda(), mask.cuda(), R=0.4, p=-1.0)

    inter = tot = 0
    for i, ev in enumerate(evs):
        ref = cluster_event_nn(ev, R=0.4, p=-1.0, dtype=np.float32)
        gp = partitions_from_jet_idx(got["jet_idx"][i].cpu().numpy(), int(got["n_jets"][i]))
        rp = partitions_from_jet_idx(ref["jet_idx"], ref["n_jets"])
        inter += len(gp & rp)
        tot += len(rp)
    assert inter / tot >= 0.97, f"{inter}/{tot} jets match"

    again = cluster_batch_triton_large(p4.cuda(), mask.cuda(), R=0.4, p=-1.0)
    for k in got:
        assert torch.equal(got[k], again[k])


def test_large_deterministic(rng):
    _, p4, mask = make_batch(rng, B=16, Nmax=300)
    p4 = p4.float().cuda()
    mask = mask.cuda()
    a = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0)
    b = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0)
    for k in a:
        assert torch.equal(a[k], b[k])


@pytest.mark.parametrize(
    "config",
    [None, (512, 4, (32, 64)), (1024, 8, (64, 128))],
    ids=["banded-default", "lowocc-b512w4", "b1024w8"],
)
@pytest.mark.parametrize("Nmax", [512, 1024])
def test_large_deterministic_matrix(rng, Nmax, config):
    """Regression guard for the kernel's ~9 hand-placed tl.debug_barrier()s.

    The scalar merge phase used to race (WAR, cross-warp) without its barrier;
    the flake only ever showed at low-occupancy launch configs AND production
    batch sizes, so this is pinned at B=256 across the launch bands and must not
    shrink to a small-B check.  Removing or reordering any barrier, or a Triton
    version changing reduction lowering, breaks bitwise determinism here.  Boosted
    (collimated) events maximize per-step stale-NN traffic, the surface most
    sensitive to the fences."""
    _, p4, mask = make_batch(rng, B=256, Nmax=Nmax, boosted=True)
    p4 = p4.float().cuda()
    mask = mask.cuda()
    kw = {} if config is None else dict(block=config[0], num_warps=config[1], tile=config[2])
    a = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0, **kw)
    for _ in range(3):
        b = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0, **kw)
        for k in a:
            assert torch.equal(a[k], b[k]), f"non-deterministic {k} (Nmax={Nmax}, config={config})"
