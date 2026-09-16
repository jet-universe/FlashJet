"""Pin the audit hardening fixes: each test encodes the failure it prevents."""

import numpy as np
import pytest
import torch

from flashjet import cluster
from flashjet.history import jet_idx_from_history
from flashjet.reference import cluster_event
from flashjet.torch_backend import cluster_batch_torch


def test_reference_survives_zero_kt2():
    """Without the kt^2 floor, a zero-pt particle at p=-1 made w=inf, the
    all-inf argmin picked the (i,i) diagonal as a 'pair' and the reference
    crashed in active.remove — the ground-truth rung could not even score
    the floored rungs on these inputs."""
    ev = np.array(
        [[0.0, 0.0, 1.0, 1.0],   # kt2 == 0, beamlike
         [3.0, 0.4, 0.2, 3.1],
         [-2.0, 1.0, -0.5, 2.4]]
    )
    seq = cluster_event(ev, R=0.4, p=-1.0)
    assert len(seq.beam_jets) == 3
    assert all(np.isfinite(h.d) for h in seq.history)

    out = cluster_batch_torch(torch.tensor(ev).unsqueeze(0).double(),
                              torch.ones(1, 3, dtype=torch.bool), R=0.4, p=-1.0)
    # build partitions from beam_jets directly: jet_constituents() applies a
    # strict pt > ptmin filter that drops the pt == 0 jet even at ptmin=0
    parts = {frozenset(seq.constituents(i)) for i in seq.beam_jets}
    got = {frozenset(np.flatnonzero(out["jet_idx"][0].numpy() == j).tolist())
           for j in range(int(out["n_jets"][0]))}
    assert got == parts


def test_history_decode_zero_width():
    """math.log2(2*N) raised ValueError at padded width N=0."""
    e = torch.full((2, 0), -2, dtype=torch.long)
    jet_idx, n_jets = jet_idx_from_history(e, e, e, torch.zeros(2, 0, dtype=torch.bool))
    assert jet_idx.shape == (2, 0)
    assert n_jets.tolist() == [0, 0]


def test_jets_p4_drops_beyond_n_jets_max():
    """jets >= n_jets_max used to be silently folded into the last slot,
    corrupting it; they must be dropped instead."""
    p4 = torch.zeros(1, 4, 4)
    p4[0, :, 0] = torch.tensor([10.0, -10.0, 5.0, -5.0])  # 4 well-separated
    p4[0, :, 1] = torch.tensor([0.0, 0.1, 8.0, -8.0])
    p4[0, :, 3] = p4[0].norm(dim=-1)
    out = cluster(p4, R=0.4, algorithm="antikt")
    assert int(out.n_jets[0]) == 4
    j2 = out.jets_p4(p4, n_jets_max=2)
    full = out.jets_p4(p4)
    assert torch.equal(j2, full[:, :2])  # slots 0-1 untouched, 2-3 dropped


def test_cluster_validates_input():
    p4 = torch.zeros(1, 3, 4)
    p4[0, :, 3] = 1.0
    bad = p4.clone()
    bad[0, 1, 3] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        cluster(bad, R=0.4)
    # the same row masked out is fine
    mask = torch.tensor([[True, False, True]])
    cluster(bad, mask=mask, R=0.4)
    # and validate=False skips the check (caller's responsibility)
    cluster(bad, R=0.4, validate=False, backend="torch")

    with pytest.raises(TypeError, match="bool"):
        cluster(p4, mask=torch.ones(1, 3, dtype=torch.int8), R=0.4)
    with pytest.raises(ValueError, match="non-finite"):
        cluster(np.array([[np.nan, 0, 0, 1.0]]), R=0.4)
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="device"):
            cluster(p4.cuda(), mask=torch.ones(1, 3, dtype=torch.bool), R=0.4)


def test_cluster_rejects_nonpositive_R():
    """R <= 0 hit three different failure modes by backend -- a host-side
    ZeroDivisionError from 1/(R*R) in the triton paths, a silent inf in
    torch/numpy -- and R < 0 was silently clustered as |R|.  cluster() now
    rejects it at the single entry point, for every backend and the numpy path."""
    ev = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    t4 = torch.tensor(ev).unsqueeze(0).float()
    m = torch.ones(1, 2, dtype=torch.bool)
    for R in (0.0, -0.4, float("nan")):
        with pytest.raises(ValueError, match="R must be positive"):
            cluster(ev, R=R)  # numpy reference path
        with pytest.raises(ValueError, match="R must be positive"):
            cluster(t4, mask=m, R=R)  # batched torch/triton path
    # the smallest positive R still clusters (guard is not over-eager)
    assert cluster(ev, R=1e-6).n_initial == 2
