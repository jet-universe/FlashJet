"""Opt-in launch-parameter autotuning for the large-N kernel.

Never on by default, on purpose: a different launch config can legitimately
flip jet_idx on near-tie merges (FMA contraction varies with codegen), so a
per-process autotuner would silently break bitwise reproducibility between
runs.  Instead the winner is benchmarked once per (GPU model, N-band) on the
caller's own batch — candidates come only from the validated sweep grid —
and persisted to a JSON cache; every later run, in any process, reads the
same config back, so outputs stay reproducible after the first tuning call.

Enable with cluster_batch_triton_large(..., tune=True) or FLASHJET_TUNE=1.
Cache: $FLASHJET_TUNE_CACHE or ~/.cache/flashjet/tune.json — delete it to
re-tune (e.g. after a triton or driver upgrade).  Concurrent tuners on one
machine are harmless: writes are atomic and either winner is valid.
"""

import json
import os
import time

import torch

# (block, num_warps, (BI, BJ)): every per-band winner measured on A100, T4,
# and H100 plus spread for other hardware.  Keep candidates inside the validated
# grid — BLOCK stays clamped at 1024 in the wrapper regardless.
CONFIGS = [
    (1024, 2, (32, 64)),
    (256, 2, (32, 64)),
    (256, 4, (32, 64)),
    (512, 4, (32, 64)),
    (512, 4, (64, 64)),
    (512, 8, (64, 128)),
    (1024, 4, (64, 128)),
    (1024, 8, (32, 64)),
    (1024, 8, (64, 64)),
    (1024, 8, (64, 128)),
    (1024, 8, (128, 128)),
]

SPIN_SECONDS = 2.5  # DVFS ramp time; cold-clock timing inverts rankings
WARMUP, TIMED = 2, 5

_proc_cache = {}


def _cache_path():
    override = os.environ.get("FLASHJET_TUNE_CACHE")
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(base, "flashjet", "tune.json")


def _load(key):
    if key in _proc_cache:
        return _proc_cache[key]
    try:
        with open(_cache_path()) as f:
            e = json.load(f)[key]
        cfg = (int(e["block"]), int(e["num_warps"]), (int(e["tile"][0]), int(e["tile"][1])))
    except (OSError, KeyError, ValueError, TypeError):
        return None
    _proc_cache[key] = cfg
    return cfg


def _store(key, cfg):
    path = _cache_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data[key] = {"block": cfg[0], "num_warps": cfg[1], "tile": list(cfg[2])}
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    _proc_cache[key] = cfg


def _spin(device):
    a = torch.randn(4096, 4096, device=device)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < SPIN_SECONDS:
        a = a @ a * 1e-3
    torch.cuda.synchronize(device)


def tuned_config(band, run, device):
    """Best (block, num_warps, tile) for this GPU and N-band.

    run(cfg) must execute the real workload with that config.  The first
    call per (device model, band) spins the clocks, benchmarks every
    candidate on it (median of TIMED runs after WARMUP, which also absorbs
    the per-config JIT compile), and persists the winner.
    """
    key = f"{torch.cuda.get_device_name(device)}|band{band}|v1"
    cfg = _load(key)
    if cfg is not None:
        return cfg
    _spin(device)
    best, best_t = None, float("inf")
    for cand in CONFIGS:
        for _ in range(WARMUP):
            run(cand)
        torch.cuda.synchronize(device)
        times = []
        for _ in range(TIMED):
            t0 = time.perf_counter()
            run(cand)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
        t = sorted(times)[TIMED // 2]
        if t < best_t:
            best, best_t = cand, t
    _store(key, best)
    return best
