"""Substructure features built on the merge history (kT & C/A).

Pins three post-clustering decoders to INDEPENDENT NumPy tree-walks on the f64
torch backend, plus physical anchors:

  * exclusive_jets_from_history  -- exclusive-kt jets by n_jets or d_cut,
    cross-checked against a prefix-of-the-sequence re-rooting and anchored to
    the inclusive partition at the trivial cut.
  * lund_coordinates_from_history -- per-split (z, dR, kt, ...) declustering
    coordinates, matched jet-for-jet to a numpy walk and tied to
    splitting_scales on the d channel.
  * groom_from_history -- soft-drop / mass-drop declustering, matched to a
    recursive numpy tagger for mMDT, soft-drop(beta=1) and the mass-drop mu.

All run on CPU (the torch backend produces the full history on CPU); no CUDA.
"""
import math

import numpy as np
import pytest
import torch

from conftest import random_event
from flashjet import cluster
from flashjet.history import (
    exclusive_jets_from_history,
    groom_from_history,
    lund_coordinates_from_history,
    splitting_scales_from_history,
)
from flashjet.kinematics import rap_phi_kt2
from flashjet.reference import cluster_event
from flashjet.torch_backend import cluster_batch_torch

ALGOS = [(1.0, "kt"), (0.0, "ca"), (-1.0, "antikt")]


def _cluster(ev, R, p):
    p4 = torch.tensor(ev[None], dtype=torch.float64)
    mask = torch.ones(1, len(ev), dtype=torch.bool)
    out = cluster_batch_torch(p4, mask, R=R, p=p)
    return out, p4, mask


def _partition(idx, n):
    idx = idx[0, :n].tolist()
    groups = {}
    for i, l in enumerate(idx):
        groups.setdefault(l, set()).add(i)
    return frozenset(frozenset(g) for g in groups.values())


# ---------------------------------------------------------------- exclusive ---

def _ref_exclusive_partition(ev, R, p, n_jets=None, d_cut=None):
    """Independent numpy exclusive partition: keep a prefix / threshold of the
    recorded pair-merges, then re-root each particle through kept edges only."""
    seq = cluster_event(ev, R=R, p=p)
    n = seq.n_initial
    pairs = [(h.parent1, h.parent2, h.child, h.d) for h in seq.history if h.parent2 != -1]
    if d_cut is not None:
        kept = [(a, b, c) for (a, b, c, d) in pairs if d < d_cut]
    else:
        k = min(len(pairs), max(0, n - max(1, n_jets)))
        kept = [(a, b, c) for (a, b, c, d) in pairs[:k]]
    up = {}
    for a, b, c in kept:
        up[a] = c
        up[b] = c

    def root(x):
        while x in up:
            x = up[x]
        return x

    groups = {}
    for i in range(n):
        groups.setdefault(root(i), set()).add(i)
    return frozenset(frozenset(g) for g in groups.values())


@pytest.mark.parametrize("p,algo", ALGOS)
def test_exclusive_matches_numpy(rng, p, algo):
    for _ in range(30):
        n = int(rng.integers(4, 20))
        ev = random_event(rng, n)
        R = 0.6
        out, p4, mask = _cluster(ev, R, p)
        n_incl = int(out["n_jets"][0])
        for target in {1, max(1, n_incl // 2), n_incl, n}:
            ei, _ = exclusive_jets_from_history(
                out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"],
                mask, n_jets=target,
            )
            assert _partition(ei, n) == _ref_exclusive_partition(ev, R, p, n_jets=target)
        dmax = float(out["hist_d"][0].max())
        for dc in (dmax * 0.1, dmax * 0.5, dmax * 1.01):
            ei, _ = exclusive_jets_from_history(
                out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"],
                mask, d_cut=dc,
            )
            assert _partition(ei, n) == _ref_exclusive_partition(ev, R, p, d_cut=dc)


@pytest.mark.parametrize("p,algo", ALGOS)
def test_exclusive_reduces_to_inclusive(rng, p, algo):
    """exclusive with n_jets == n_inclusive reproduces the inclusive partition."""
    n = 14
    ev = random_event(rng, n)
    R = 0.6
    out, p4, mask = _cluster(ev, R, p)
    n_incl = int(out["n_jets"][0])
    ei, ne = exclusive_jets_from_history(
        out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"],
        mask, n_jets=n_incl,
    )
    assert int(ne[0]) == n_incl
    assert _partition(ei, n) == _partition(out["jet_idx"], n)


def test_exclusive_requires_one_cut(rng):
    out, p4, mask = _cluster(random_event(rng, 8), 0.6, 1.0)
    args = (out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"], mask)
    with pytest.raises(ValueError):
        exclusive_jets_from_history(*args)
    with pytest.raises(ValueError):
        exclusive_jets_from_history(*args, n_jets=2, d_cut=0.1)


def test_exclusive_via_cluster_output(rng):
    """The ClusterOutput wrapper carries the mask and matches the raw helper."""
    n = 10
    ev = random_event(rng, n)
    p4 = torch.tensor(ev[None], dtype=torch.float64)
    mask = torch.ones(1, n, dtype=torch.bool)
    co = cluster(p4, mask, R=0.6, algorithm="kt", backend="torch")
    assert co.mask is not None
    ei, ne = co.exclusive_jets(n_jets=3)
    ref, _ = exclusive_jets_from_history(
        co.hist_p1, co.hist_p2, co.hist_child, co.hist_d, mask, n_jets=3,
    )
    assert torch.equal(ei, ref)


# --------------------------------------------------------------------- lund ---

def _ref_lund(ev, R, p):
    seq = cluster_event(ev, R=R, p=p)
    P = seq.p4
    children = {h.child: (h.parent1, h.parent2, h.d) for h in seq.history if h.child != -1}
    jetof = {}
    for jnum, jid in enumerate(seq.beam_jets):
        stack = [jid]
        while stack:
            x = stack.pop()
            if x in children:
                jetof[x] = jnum
                a, b, _ = children[x]
                stack += [a, b]
    per = {}
    for c in sorted(children):
        a, b, d = children[c]
        j = jetof.get(c)
        if j is None:
            continue
        pa, pb = P[a], P[b]
        ra, pha, k2a = rap_phi_kt2(*[np.array([v]) for v in pa])
        rb, phb, k2b = rap_phi_kt2(*[np.array([v]) for v in pb])
        pta, ptb = float(np.sqrt(max(k2a[0], 0))), float(np.sqrt(max(k2b[0], 0)))
        dphi = abs(pha[0] - phb[0])
        dphi = min(dphi, 2 * np.pi - dphi)
        dR = float(np.sqrt((ra[0] - rb[0]) ** 2 + dphi ** 2))
        ptmin = min(pta, ptb)
        per.setdefault(j, []).append((ptmin / max(pta + ptb, 1e-30), dR, ptmin * dR, d))
    return {j: v[::-1] for j, v in per.items()}  # de-clustering order


@pytest.mark.parametrize("p,algo", ALGOS)
def test_lund_matches_numpy(rng, p, algo):
    for _ in range(25):
        n = int(rng.integers(4, 18))
        ev = random_event(rng, n, boosted=True)
        R = 0.8
        out, p4, mask = _cluster(ev, R, p)
        L = lund_coordinates_from_history(
            out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"],
            mask, p4, R,
        )[0]
        ss = splitting_scales_from_history(
            out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"],
        )[0]
        # channel 5 == splitting_scales (exact tie)
        assert torch.allclose(L[..., 5], ss, atol=1e-12)
        J, S, _ = L.shape

        def sig(rows):
            return tuple((round(z, 7), round(dR, 7), round(kt, 7), round(d, 7)) for z, dR, kt, d in rows)

        ref = sorted(sig(v) for v in _ref_lund(ev, R, p).values())
        impl = []
        for j in range(J):
            rows = [(L[j, s, 0].item(), L[j, s, 1].item(), L[j, s, 2].item(), L[j, s, 5].item())
                    for s in range(S) if L[j, s, 5].item() != 0]
            if rows:
                impl.append(sig(rows))
        assert sorted(impl) == ref
        # z physical bounds on real splits
        zc = L[..., 0][ss > 0]
        if zc.numel():
            assert float(zc.min()) > 0 and float(zc.max()) <= 0.5 + 1e-9


# ------------------------------------------------------------------- groom ----

def _mass(pv):
    return float(np.sqrt(max(pv[3] ** 2 - pv[0] ** 2 - pv[1] ** 2 - pv[2] ** 2, 0)))


def _ref_groom(ev, R, p, z_cut, beta, mu=None):
    seq = cluster_event(ev, R=R, p=p)
    P = seq.p4
    children = {h.child: (h.parent1, h.parent2) for h in seq.history if h.child != -1}
    res = []
    for jid in seq.beam_jets:
        cur = jid
        tagged, gp4, zz, dd = False, np.zeros(4), 0.0, 0.0
        while cur in children:
            a, b = children[cur]
            pa, pb = P[a], P[b]
            ra, pha, k2a = rap_phi_kt2(*[np.array([v]) for v in pa])
            rb, phb, k2b = rap_phi_kt2(*[np.array([v]) for v in pb])
            pti, ptj = float(np.sqrt(max(k2a[0], 0))), float(np.sqrt(max(k2b[0], 0)))
            dphi = abs(pha[0] - phb[0])
            dphi = min(dphi, 2 * np.pi - dphi)
            dR = float(np.sqrt((ra[0] - rb[0]) ** 2 + dphi ** 2))
            z = min(pti, ptj) / max(pti + ptj, 1e-30)
            ok = z > z_cut * (dR / R) ** beta
            if mu is not None:
                ok = ok and (max(_mass(pa), _mass(pb)) < mu * max(_mass(P[cur]), 1e-30))
            if ok:
                tagged, gp4, zz, dd = True, P[cur].copy(), z, dR
                break
            cur = a if pti >= ptj else b
        res.append((tagged, gp4, zz, dd))
    return res


@pytest.mark.parametrize("p,algo", ALGOS)
@pytest.mark.parametrize("z_cut,beta,mu", [(0.1, 0.0, None), (0.05, 1.0, None), (0.09, 0.0, 0.67)])
def test_groom_matches_numpy(rng, p, algo, z_cut, beta, mu):
    for _ in range(20):
        n = int(rng.integers(3, 18))
        ev = random_event(rng, n, boosted=True)
        R = 0.8
        out, p4, mask = _cluster(ev, R, p)
        g = groom_from_history(
            out["hist_p1"], out["hist_p2"], out["hist_child"], out["hist_d"],
            mask, p4, R, z_cut=z_cut, beta=beta, mu=mu,
        )
        ref = _ref_groom(ev, R, p, z_cut, beta, mu)
        J = g["tagged"].shape[1]

        def imp():
            return sorted(
                (bool(g["tagged"][0, j]),
                 tuple(round(x, 6) for x in g["groomed_p4"][0, j].tolist()),
                 round(float(g["z"][0, j]), 6), round(float(g["dR"][0, j]), 6))
                for j in range(J)
            )

        def rf():
            return sorted(
                (tg, tuple(round(float(x), 6) for x in gp), round(zz, 6), round(dd, 6))
                for tg, gp, zz, dd in ref
            )

        assert imp() == rf()


@pytest.mark.parametrize("p,algo", ALGOS)
def test_groomed_mass_not_larger(rng, p, algo):
    """A tagged groomed subjet has mass <= its ungroomed jet mass."""
    n = 16
    ev = random_event(rng, n, boosted=True)
    R = 0.8
    p4 = torch.tensor(ev[None], dtype=torch.float64)
    mask = torch.ones(1, n, dtype=torch.bool)
    co = cluster(p4, mask, R=R, p=p, backend="torch")
    jets = co.jets_p4(p4)
    g = co.groomed_jets(p4, R, z_cut=0.1, beta=0.0)
    for j in range(g["tagged"].shape[1]):
        if bool(g["tagged"][0, j]):
            gm = _mass(g["groomed_p4"][0, j].tolist())
            jm = _mass(jets[0, j].tolist())
            assert gm <= jm + 1e-6
