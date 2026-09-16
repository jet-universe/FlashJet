"""Kinematic helpers shared by the NumPy reference and the torch backend.

All formulas follow FastJet's PseudoJet conventions:
  * rapidity (not pseudorapidity) is used in the kt-family distance,
    computed with FastJet's numerically stable form
        y = -sign(pz) * 0.5 * log((kt2 + m2_eff) / (E + |pz|)^2)
    with m2_eff = max(0, m^2), and a MaxRap guard for beam-like particles.
  * delta-phi is folded into [0, pi].
"""

import numpy as np

MAX_RAP = 1e5


def rap_phi_kt2(px, py, pz, E, xp=np):
    """Return (rapidity, phi, kt2) for arrays of momentum components.

    ``xp`` is the array module: numpy or torch (the ops used are common to
    both).
    """
    kt2 = px * px + py * py
    phi = xp.arctan2(py, px) if xp is np else xp.atan2(py, px)
    m2_eff = E * E - pz * pz - kt2
    m2_eff = xp.maximum(m2_eff, xp.zeros_like(m2_eff))
    abs_pz = xp.abs(pz)
    denom = (E + abs_pz) ** 2
    beamlike = (kt2 + m2_eff) <= 0
    ratio = xp.where(beamlike, xp.ones_like(denom), (kt2 + m2_eff) / denom)
    half_log = 0.5 * xp.log(ratio)  # <= 0
    rap = xp.where(pz >= 0, -half_log, half_log)
    rap = xp.where(beamlike, xp.where(pz >= 0, MAX_RAP + abs_pz, -(MAX_RAP + abs_pz)), rap)
    return rap, phi, kt2
