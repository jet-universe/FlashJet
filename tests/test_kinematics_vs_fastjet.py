"""Jet-by-jet kinematics (pt, y, phi, m) agreement with real FastJet.

Two levels:
  * float64 torch backend must match FastJet essentially exactly;
  * float32 (what the GPU kernels use) must match to f32 rounding on all
    matched jets, with only rare near-tie events differing in multiplicity.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
fastjet = pytest.importorskip("fastjet")

from flashjet.torch_backend import cluster_batch_torch
from test_torch_vs_reference import make_batch

R = 0.4


def fastjet_jets(ev):
    jd = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    pjs = [fastjet.PseudoJet(*map(float, row)) for row in ev]
    cs = fastjet.ClusterSequence(pjs, jd)
    jets = fastjet.sorted_by_pt(cs.inclusive_jets())
    return np.array([[j.px(), j.py(), j.pz(), j.E()] for j in jets]).reshape(-1, 4)


def flashjet_jets(out, p4, b):
    nj = int(out["n_jets"][b])
    jets = np.zeros((nj, 4))
    for j in range(nj):
        sel = out["jet_idx"][b] == j
        jets[j] = p4[b][sel].sum(0).numpy()
    order = np.argsort(-np.hypot(jets[:, 0], jets[:, 1]), kind="stable")
    return jets[order]


def kin(jets):
    px, py, pz, E = jets.T
    pt = np.hypot(px, py)
    y = 0.5 * np.log((E + pz) / (E - pz))
    phi = np.arctan2(py, px)
    m2 = E**2 - px**2 - py**2 - pz**2
    return pt, y, phi, m2


@pytest.mark.parametrize("dtype,pt_rtol,ang_atol,frac", [
    (torch.float64, 1e-9, 1e-9, 1.0),
    (torch.float32, 2e-4, 2e-4, 0.95),
])
def test_jet_kinematics_match_fastjet(rng, dtype, pt_rtol, ang_atol, frac):
    events, p4, mask = make_batch(rng, B=48, Nmax=80)
    out = cluster_batch_torch(p4.to(dtype), mask, R=R, p=-1.0)

    matched = 0
    for b, ev in enumerate(events):
        fj = fastjet_jets(ev)
        mine = flashjet_jets(out, p4.to(dtype), b)
        if len(fj) != len(mine):
            continue  # near-tie merge flip (float32 only); counted below
        matched += 1
        pt_f, y_f, phi_f, m2_f = kin(fj)
        pt_m, y_m, phi_m, m2_m = kin(mine)
        np.testing.assert_allclose(pt_m, pt_f, rtol=pt_rtol)
        np.testing.assert_allclose(y_m, y_f, atol=ang_atol)
        dphi = np.abs(phi_m - phi_f)
        assert (np.minimum(dphi, 2 * np.pi - dphi) < ang_atol).all()
        # m^2 = E^2 - p^2 is cancellation-prone: the float32 floor is set by
        # rounding of the big components, i.e. an absolute error ~ eps * E^2.
        # (If precise light-jet masses matter, sum in f64: jets_p4(p4.double()))
        eps = 2e-5 if dtype == torch.float32 else 1e-12
        assert (np.abs(m2_m - m2_f) <= eps * fj[:, 3] ** 2 + 1e-9).all()
    assert matched >= frac * len(events)
