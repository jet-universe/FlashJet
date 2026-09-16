#!/usr/bin/env python
"""Sweep launch parameters of the large-N kernel.  Usage: tune_large.py [B] [N]"""

import itertools
import sys
import time

import torch

from flashjet.triton_large import cluster_batch_triton_large

sys.path.insert(0, "tests")
from conftest import random_event  # noqa: E402
import numpy as np  # noqa: E402


def make(B, N):
    rng = np.random.default_rng(7)
    p4 = torch.zeros(B, N, 4)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        n = int(rng.integers(N // 2, N + 1))
        p4[b, :n] = torch.from_numpy(random_event(rng, n)).float()
        mask[b, :n] = True
    return p4.cuda(), mask.cuda()


def spin_clocks(seconds=2.5):
    """Hold the GPU busy until DVFS reaches boost clocks; without this the
    first configs of the sweep are timed up to ~1.9x slow and the ranking
    inverts (clocks ramp 210 -> 1410 MHz over ~1-2 s of sustained load)."""
    a = torch.randn(4096, 4096, device="cuda")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ a * 1e-3
    torch.cuda.synchronize()


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    p4, mask = make(B, N)
    ref = None
    # note: jet_idx can legitimately differ across configs for near-tie
    # merges (FMA contraction varies with codegen); [DIFF] is informational
    for blk, nw, tile in itertools.product(
        (256, 512, 1024), (2, 4, 8), ((32, 64), (64, 64), (64, 128), (128, 128))
    ):
        try:
            fn = lambda: cluster_batch_triton_large(p4, mask, 0.4, -1.0, block=blk, num_warps=nw, tile=tile)
            out = fn()
            spin_clocks()
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            dt = sorted(times)[len(times) // 2]
            if ref is None:
                ref = out["jet_idx"]
            ok = "ok " if torch.equal(out["jet_idx"], ref) else "DIFF"
            print(f"block={blk:5d} warps={nw} tile={tile}: {dt*1e3:8.1f} ms/batch  {dt/B*1e6:7.1f} us/event  [{ok}]")
        except Exception as e:  # noqa: BLE001
            print(f"block={blk:5d} warps={nw} tile={tile}: FAILED {type(e).__name__}")


if __name__ == "__main__":
    main()
