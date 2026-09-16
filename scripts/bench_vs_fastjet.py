#!/usr/bin/env python
"""Benchmark flashjet against scikit-hep fastjet (CPU reference binder).

Runs whatever is available in the current environment:
  * fastjet classic interface  (per-event python loop over PseudoJets)
  * fastjet awkward interface  (vectorized multi-event, still CPU)
  * flashjet cpu     (batched NumPy NN strategy, the CPU default)
  * flashjet torch (CPU and/or CUDA)
  * flashjet triton / triton-large (CUDA)

Usage:  python scripts/bench_vs_fastjet.py [B] [N] [iters]
e.g.    python scripts/bench_vs_fastjet.py 256 64      # jet-level reclustering
        python scripts/bench_vs_fastjet.py 32 6000 3   # full-event clustering
"""

import sys
import time

import numpy as np

R, P = 0.4, -1.0  # anti-kt


def gen_events(B, N, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        n = int(rng.integers(max(N // 2, 1), N + 1))
        pt = rng.uniform(0.5, 80.0, n)
        y = rng.uniform(-3.0, 3.0, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        m = rng.uniform(0.0, 1.0, n)
        mt = np.sqrt(m**2 + pt**2)
        out.append(
            np.stack([pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y), mt * np.cosh(y)], 1)
        )
    return out


def timeit(fn, iters):
    fn()  # warmup (also triggers JIT/compile where applicable)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    events = gen_events(B, N)
    n_tot = sum(len(e) for e in events)
    print(f"anti-kt R={R}: {B} events, <=N={N} constituents ({n_tot} particles total)\n")
    rows = []

    import fastjet

    jd = fastjet.JetDefinition(fastjet.antikt_algorithm, R)

    def run_fj_classic():
        for ev in events:
            pjs = [fastjet.PseudoJet(*map(float, row)) for row in ev]
            cs = fastjet.ClusterSequence(pjs, jd)
            cs.inclusive_jets()

    rows.append(("fastjet classic (CPU loop)", timeit(run_fj_classic, iters)))

    try:
        import awkward as ak

        ak_events = ak.Array(
            [[{"px": r[0], "py": r[1], "pz": r[2], "E": r[3]} for r in ev] for ev in events]
        )

        def run_fj_awkward():
            cs = fastjet.ClusterSequence(ak_events, jd)
            cs.inclusive_jets()

        rows.append(("fastjet awkward (CPU vect.)", timeit(run_fj_awkward, iters)))
    except ImportError:
        print("awkward not installed -> skipping fastjet awkward interface")

    import torch

    import flashjet
    from flashjet.cpu_backend import cluster_batch_cpu

    p4 = torch.zeros(B, N, 4)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b, ev in enumerate(events):
        p4[b, : len(ev)] = torch.from_numpy(ev).float()
        mask[b, : len(ev)] = True

    from flashjet import _native

    if _native.HAS_NATIVE:
        import os

        nt = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        rows.append(
            (f"flashjet cpu C++ ({nt} threads)",
             timeit(lambda: flashjet.cluster(p4, mask, R=R, p=P, backend="cpu"), iters))
        )
        rows.append(
            ("flashjet cpu C++ (1 thread)",
             timeit(lambda: cluster_batch_cpu(p4, mask, R=R, p=P, threads=1), iters))
        )
    else:
        print("C++ kernel not built -> NumPy fallback only (pip install -e . to build it)")
    rows.append(
        ("flashjet cpu (NumPy fallback)",
         timeit(lambda: cluster_batch_cpu(p4, mask, R=R, p=P, native=False), iters))
    )

    if N <= 128:
        rows.append(
            ("flashjet torch (CPU, O(N^3))",
             timeit(lambda: flashjet.cluster(p4, mask, R=R, p=P, backend="torch"), iters))
        )
    else:
        print(f"N={N} > 128 -> skipping O(N^3) torch backend on CPU")

    if torch.cuda.is_available():
        p4c, maskc = p4.cuda(), mask.cuda()

        def gpu(fn):
            def g():
                fn()
                torch.cuda.synchronize()
            return g

        if N <= 256:
            rows.append(
                ("flashjet torch (GPU)",
                 timeit(gpu(lambda: flashjet.cluster(p4c, maskc, R=R, p=P, backend="torch")), iters))
            )
        try:
            if N <= 128:
                rows.append(
                    ("flashjet triton fused (GPU)",
                     timeit(gpu(lambda: flashjet.cluster(p4c, maskc, R=R, p=P, backend="triton")), iters))
                )
            rows.append(
                ("flashjet triton-large (GPU)",
                 timeit(gpu(lambda: flashjet.cluster(p4c, maskc, R=R, p=P, backend="triton-large")), iters))
            )
        except Exception as e:  # noqa: BLE001
            print(f"triton backends failed: {e}")
    else:
        print("no CUDA -> GPU backends skipped (run this on the training machine)")

    rows = [r for r in rows if r and r[0]]
    base = rows[0][1]
    print(f"\n{'backend':<32} {'ms/batch':>10} {'us/event':>10} {'speedup':>8}")
    for name, t in rows:
        print(f"{name:<32} {t*1e3:>10.2f} {t/B*1e6:>10.1f} {base/t:>7.1f}x")


if __name__ == "__main__":
    main()
