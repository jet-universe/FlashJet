"""Per-jet splitting scales (substructure features) + optional-decode.

Pins ClusterOutput.splitting_scales / history.splitting_scales_from_history to
an INDEPENDENT NumPy tree-walk (naive chain-following, a different algorithm
from the vectorized pointer-jump + one-hot impl), exactly, on the f64 torch
backend; plus a physical anchor (Cambridge/Aachen d_min is monotonic, so the
de-clustering sequence must be descending) and jet-ordering alignment with
jets_p4.  The decode=False path is checked on CUDA against decode=True.
"""
import numpy as np
import pytest
import torch

from conftest import random_event
from flashjet import cluster
from flashjet.history import splitting_scales_from_history
from flashjet.torch_backend import cluster_batch_torch


def _ref_scales(hp1, hp2, hch, hd, J):
    """Naive per-event tree-walk: for each pair merge follow its child up the
    merge tree to the beam step that finalises its jet, group d in clustering
    order, reverse to de-clustering order, pad to (B, J, S)."""
    hp1, hp2, hch, hd = (np.asarray(x) for x in (hp1, hp2, hch, hd))
    B, N = hp1.shape
    rows = []  # rows[b][j] = list of d in de-clustering order
    S = 1
    for b in range(B):
        beam_jet = {}      # step -> jet number (beam steps only)
        consumer = {}      # pseudojet id -> the step that consumes it as a parent
        bc = 0
        for s in range(N):
            if hp2[b, s] == -1:        # beam merge
                beam_jet[s] = bc; bc += 1
                consumer[int(hp1[b, s])] = s
            elif hp2[b, s] >= 0:       # pair merge
                consumer[int(hp1[b, s])] = s
                consumer[int(hp2[b, s])] = s
        jets = {j: [] for j in range(J)}
        for s in range(N):
            if hp2[b, s] < 0:
                continue
            cur = int(hch[b, s])
            jet = -1
            while cur in consumer:
                cs = consumer[cur]
                if hp2[b, cs] == -1:
                    jet = beam_jet[cs]; break
                cur = int(hch[b, cs])
            if 0 <= jet < J:
                jets[jet].append((s, hd[b, s]))
        ev = []
        for j in range(J):
            ds = [d for _, d in sorted(jets[j])][::-1]  # clustering -> de-clustering
            ev.append(ds); S = max(S, len(ds))
        rows.append(ev)
    out = np.zeros((B, J, S), dtype=np.float64)
    for b in range(B):
        for j in range(J):
            out[b, j, : len(rows[b][j])] = rows[b][j]
    return out


def _batch(B, n_lo, n_hi, seed=0):
    rng = np.random.default_rng(seed)
    N = n_hi
    p4 = torch.zeros(B, N, 4, dtype=torch.float64)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        n = int(rng.integers(n_lo, n_hi + 1))
        p4[b, :n] = torch.from_numpy(random_event(rng, n)).double()
        mask[b, :n] = True
    return p4, mask


@pytest.mark.parametrize("p,name", [(0.0, "ca"), (1.0, "kt"), (-1.0, "antikt")])
def test_grouping_matches_treewalk(p, name):
    """Exact: the vectorized grouping/ranking equals the naive tree-walk."""
    p4, mask = _batch(6, 8, 16, seed=11)
    out = cluster_batch_torch(p4, mask, R=0.5, p=p)
    hp1, hp2, hch, hd = out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"]
    J = int((hp2 == -1).sum(1).max())
    mine = splitting_scales_from_history(hp1, hp2, hch, hd).numpy()
    ref = _ref_scales(hp1, hp2, hch, hd, J)
    assert mine.shape == ref.shape, (mine.shape, ref.shape)
    assert np.array_equal(mine, ref)


def test_kt_scales_descending():
    """Physical anchor: kt d_min is monotonic, so each jet's de-clustering
    sequence (incl. zero padding) is non-increasing -- the exclusive
    d_12 >= d_23 >= ... scales.  (C/A and anti-kt d_min are NOT monotonic, so
    this ordering is kt-specific.)"""
    p4, mask = _batch(8, 6, 20, seed=3)
    out = cluster_batch_torch(p4, mask, R=0.6, p=1.0)
    sc = splitting_scales_from_history(
        out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"]
    )
    assert (sc[..., :-1] >= sc[..., 1:] - 1e-12).all()
    assert (sc >= 0).all()


def test_single_jet_counts_and_top():
    """A collimated spray (all inside a small cone) clusters to ONE jet: k
    leaves -> k-1 binary merges, and for kt slot 0 is the hardest (d_12)."""
    rng = np.random.default_rng(7)
    k = 12
    pt = torch.tensor(rng.uniform(2.0, 50.0, k))
    y = torch.tensor(rng.uniform(-0.1, 0.1, k))        # tight cone, dR << R
    phi = torch.tensor(rng.uniform(-0.1, 0.1, k))
    m = torch.tensor(rng.uniform(0.0, 1.0, k))
    mt = (m**2 + pt**2).sqrt()
    p4 = torch.stack(
        [pt * phi.cos(), pt * phi.sin(), mt * torch.sinh(y), mt * torch.cosh(y)], -1
    )[None].double()
    mask = torch.ones(1, k, dtype=torch.bool)
    out = cluster_batch_torch(p4, mask, R=1.0, p=1.0)  # kt; cone << R -> 1 jet
    assert int(out["n_jets"][0]) == 1
    sc = splitting_scales_from_history(
        out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"]
    )
    assert sc.shape == (1, 1, k - 1)            # k leaves -> k-1 binary merges
    assert (sc[0, 0] > 0).sum() == k - 1        # all real, none padded
    assert sc[0, 0, 0] == sc[0, 0].max()        # slot 0 is d_12 (largest, kt)


def test_alignment_with_jets_p4_and_api():
    """splitting_scales() (via the public API) lines up jet-for-jet with
    jets_p4: same J, and n_jets_max truncates both the same way."""
    p4, mask = _batch(5, 10, 24, seed=5)
    res = cluster(p4, mask, R=0.4, algorithm="cambridge", backend="torch")
    jp4 = res.jets_p4(p4)
    sc = res.splitting_scales()
    assert sc.shape[0] == jp4.shape[0] and sc.shape[1] == jp4.shape[1]
    sc2 = res.splitting_scales(n_jets_max=2)
    assert sc2.shape[1] == 2
    # truncated output is the first two jets of the full one (beam-merge order)
    assert np.array_equal(sc2.numpy(), sc.numpy()[:, :2, : sc2.shape[2]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_decode_optional_triton_large():
    """decode=False skips the per-particle decode: jet_idx is None, n_jets and
    the full merge history are bitwise-identical, and the scales agree."""
    from flashjet.triton_large import cluster_batch_triton_large

    p4, mask = _batch(64, 40, 64, seed=9)
    p4, mask = p4.float().cuda(), mask.cuda()
    full = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0, decode=True)
    nodec = cluster_batch_triton_large(p4, mask, R=0.4, p=-1.0, decode=False)
    assert nodec["jet_idx"] is None
    assert torch.equal(full["n_jets"], nodec["n_jets"])
    for k in ("hist_p1", "hist_p2", "hist_child", "hist_d"):
        assert torch.equal(full[k], nodec[k]), k
    a = splitting_scales_from_history(*[full[k] for k in
        ("hist_p1", "hist_p2", "hist_child", "hist_d")])
    b = splitting_scales_from_history(*[nodec[k] for k in
        ("hist_p1", "hist_p2", "hist_child", "hist_d")])
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_jets_p4_errors_without_decode():
    out = cluster(_batch(4, 20, 33, seed=2)[0].float().cuda(),
                  _batch(4, 20, 33, seed=2)[1].cuda(),
                  R=0.4, algorithm="antikt", decode=False)
    assert out.jet_idx is None
    with pytest.raises(ValueError, match="decode=True"):
        out.jets_p4(torch.zeros(4, 33, 4, device="cuda"))
