"""Pin the pair-vs-beam EXACT-tie contract to real FastJet's semantics.

FastJet seeds every pseudojet's NN search with the beam at dist = R^2 and
lets a pair displace it only on strictly smaller distance
(ClusterSequence.hh, _bj_set_NN_nocross): an exact d_ij == d_iB tie is a
BEAM merge.  The events below are built from massless axis-aligned
particles (exact rapidity 0, exact phi in {0, pi/2, pi}) with R chosen so
the tie d_ij == d_iB == 1.0 is exact in the computed float64 values of
BOTH op-order forms in use: the divide form (reference, torch) and the
reciprocal-multiply form (nn_reference).  Without this pin, the rungs can
silently disagree on adversarial input while every random-event test stays
green (ties have measure zero there).
"""

import math

import numpy as np
import pytest
import torch

from flashjet.nn_reference import cluster_event_nn
from flashjet.reference import BEAM, cluster_event
from flashjet.torch_backend import cluster_batch_torch

# event -> (p4 rows, R); p = 0 (C/A) makes every d_iB exactly 1.0
TIE_EVENTS = {
    "back_to_back": ([[1.0, 0, 0, 1.0], [-1.0, 0, 0, 1.0]], math.pi),
    "three_axes": ([[1.0, 0, 0, 1.0], [0, 1.0, 0, 1.0], [-1.0, 0, 0, 1.0]], math.pi / 2),
}


@pytest.mark.parametrize("name", sorted(TIE_EVENTS))
def test_tie_is_exact_in_both_op_orders(name):
    p4, R = TIE_EVENTS[name]
    a = np.asarray(p4, dtype=np.float64)
    dphi = abs(math.atan2(a[1, 1], a[1, 0]) - math.atan2(a[0, 1], a[0, 0]))
    dr2 = min(dphi, 2 * math.pi - dphi) ** 2
    assert dr2 / (R * R) == 1.0          # divide form (reference, torch)
    assert dr2 * (1.0 / (R * R)) == 1.0  # reciprocal form (nn_reference)


@pytest.mark.parametrize("name", sorted(TIE_EVENTS))
def test_all_rungs_take_the_beam_branch(name):
    p4, R = TIE_EVENTS[name]
    ev = np.asarray(p4, dtype=np.float64)
    n = len(ev)

    seq = cluster_event(ev, R=R, p=0.0)
    assert len(seq.beam_jets) == n
    assert seq.history[0].parent2 == BEAM

    nn = cluster_event_nn(ev, R=R, p=0.0)
    assert nn["n_jets"] == n
    assert nn["hist_p2"][0] == BEAM

    t4 = torch.tensor(ev, dtype=torch.float64).unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool)
    out = cluster_batch_torch(t4, mask, R=R, p=0.0)
    assert out["n_jets"].tolist() == [n]
    assert out["hist_p2"][0, 0].item() == BEAM
    # all-beam ties also pin the slot order: lowest slot merges first
    assert out["jet_idx"][0].tolist() == list(range(n))

    if torch.cuda.is_available():
        out_gpu = cluster_batch_torch(t4.cuda(), mask.cuda(), R=R, p=0.0)
        for k in ("jet_idx", "n_jets", "hist_p1", "hist_p2", "hist_child"):
            assert torch.equal(out_gpu[k].cpu(), out[k])


@pytest.mark.parametrize("name", sorted(TIE_EVENTS))
def test_kernels_take_the_beam_branch(name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    pytest.importorskip("triton")
    from flashjet.triton_backend import cluster_batch_triton
    from flashjet.triton_large import cluster_batch_triton_large

    p4, R = TIE_EVENTS[name]
    n = len(p4)
    t4 = torch.tensor(p4, dtype=torch.float32, device="cuda").unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    for fn in (cluster_batch_triton, cluster_batch_triton_large):
        out = fn(t4, mask, R=R, p=0.0)
        assert out["n_jets"].tolist() == [n]


@pytest.mark.parametrize("p", [-1.0, 1.0])
def test_kinematics_parity_kernels_vs_reference(p):
    """Pin the three hand-transcribed copies of the kinematics conventions
    (kinematics.py for reference+torch; inline in triton_backend._cluster_kernel
    and triton_large._rap_w) against each other on a crafted event.

    The >=0.97/0.98 match-fraction kernel tests cannot catch a small constant
    drift in one copy -- e.g. the 1e-30 kt^2 floor or the MaxRap guard -- that
    perturbs only a few percent of merges.  This event is strictly separated (a
    zero-pt beamlike particle, one pair at dR << R, one isolated particle), so it
    has a unique merge order in every rung: each f32 kernel must reproduce the
    f64 reference partition EXACTLY and match its sorted merge distances within
    f32 tolerance.  The beamlike particle's distance is w = max(kt2, 1e-30)**p,
    set entirely by the floor (1e30 at p=-1, 1e-30 at p=1), so a floor drift in
    either kernel -- invisible to the match-fraction tests -- fails here."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    pytest.importorskip("triton")
    from flashjet.triton_backend import cluster_batch_triton
    from flashjet.triton_large import cluster_batch_triton_large
    from test_nn_vs_reference import partitions_from_jet_idx

    ev = np.array(
        [[0.0, 0.0, 50.0, 50.0],     # zero-pt beamlike: kt2 == 0, distance is pure floor
         [20.0, 0.0, 0.0, 20.0],     # ┐ one pair, dR ~ 0.1 << R (unambiguous merge)
         [19.9, 2.0, 0.0, 20.0],     # ┘
         [-18.0, -3.0, 6.0, 19.5]],  # one isolated particle (beam merge)
        dtype=np.float64,
    )
    n = len(ev)
    seq = cluster_event(ev, R=0.4, p=p)
    ref_parts = {frozenset(seq.constituents(i)) for i in seq.beam_jets}
    ref_d = np.sort([h.d for h in seq.history])
    assert len(ref_parts) == 3  # {beamlike}, {pair}, {isolated}: a non-degenerate partition

    t4 = torch.tensor(ev, dtype=torch.float32, device="cuda").unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    for fn in (cluster_batch_triton, cluster_batch_triton_large):
        out = fn(t4, mask, R=0.4, p=p)
        got_parts = partitions_from_jet_idx(out["jet_idx"][0].cpu().numpy(), int(out["n_jets"][0]))
        assert got_parts == ref_parts, f"{fn.__name__} p={p}: partition drift from reference"
        got_d = np.sort(out["hist_d"][0, :n].cpu().numpy())
        np.testing.assert_allclose(
            got_d, ref_d, rtol=1e-3,
            err_msg=f"{fn.__name__} p={p}: merge-distance drift (a kinematics constant?)",
        )
