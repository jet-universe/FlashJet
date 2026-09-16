import numpy as np
import pytest


def random_event(rng, n, boosted=False):
    """Physically sensible massive particles: sample pt, y, phi, m."""
    pt = rng.uniform(0.5, 80.0, n)
    y = rng.uniform(-3.0, 3.0, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    m = rng.uniform(0.0, 1.0, n)
    if boosted:  # collimated spray to exercise dense merging
        y = rng.normal(0.5, 0.3, n)
        phi = np.mod(rng.normal(1.0, 0.3, n), 2 * np.pi)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    mt = np.sqrt(m**2 + pt**2)
    pz = mt * np.sinh(y)
    E = mt * np.cosh(y)
    return np.stack([px, py, pz, E], axis=1)


@pytest.fixture
def rng():
    return np.random.default_rng(20260610)
