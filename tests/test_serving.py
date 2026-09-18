"""Serving helpers: the NumPy history decode and cluster_events.

These run without torch: the decode is plain NumPy, and cluster_events falls
back to the C++ kernel or the NumPy reference when no GPU is around.
"""

import numpy as np
import pytest

from flashjet import jet_idx_from_history_np
from flashjet.nn_reference import cluster_event_nn
from flashjet.serving import MAX_PARTICLES, cluster_events, resolve_backend


def random_events(rng, counts):
    events = []
    for n in counts:
        pt = 0.1 + rng.exponential(5.0, n)
        eta = rng.uniform(-4.0, 4.0, n)
        phi = rng.uniform(-np.pi, np.pi, n)
        pz = pt * np.sinh(eta)
        events.append(np.stack([pt * np.cos(phi), pt * np.sin(phi), pz, np.hypot(pt, pz)], axis=1))
    return events


@pytest.mark.parametrize("p", [-1.0, 0.0, 1.0])
def test_numpy_decode_matches_reference(p):
    rng = np.random.default_rng(7)
    for event in random_events(rng, [1, 2, 5, 40, 200]):
        out = cluster_event_nn(event, R=0.4, p=p)
        jet_idx, n_jets = jet_idx_from_history_np(
            out["hist_p1"], out["hist_p2"], out["hist_child"], len(event)
        )
        assert n_jets == out["n_jets"]
        assert np.array_equal(jet_idx, out["jet_idx"])


@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_cluster_events_matches_reference(backend):
    rng = np.random.default_rng(11)
    events = random_events(rng, [3, 17, 60, 150])
    results = cluster_events(events, R=0.4, p=-1.0, backend=backend)
    assert len(results) == len(events)
    for event, (jet_idx, n_jets) in zip(events, results):
        reference = cluster_event_nn(event, R=0.4, p=-1.0)
        assert n_jets == reference["n_jets"]
        assert np.array_equal(jet_idx, reference["jet_idx"])


def test_cluster_events_empty_and_oversized():
    assert cluster_events([]) == []
    with pytest.raises(ValueError):
        cluster_events([np.zeros((MAX_PARTICLES + 1, 4))])


def test_resolve_backend():
    assert resolve_backend("auto") in ("gpu", "native", "numpy")
    assert resolve_backend("numpy") == "numpy"
    with pytest.raises(ValueError):
        resolve_backend("nonsense")


def test_model_repository(tmp_path):
    from flashjet.serving.model_repository import write

    model_dir = write(str(tmp_path), name="flashjet")
    config = open(f"{model_dir}/config.pbtxt").read()
    assert 'name: "flashjet"' in config and "dynamic_batching" in config
    assert "TritonPythonModel" in open(f"{model_dir}/1/model.py").read()
