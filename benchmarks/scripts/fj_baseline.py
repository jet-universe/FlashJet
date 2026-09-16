"""FastJet CPU baseline for the standalone event-processing bench.

Uses the SAME event generator as scripts/bench_vs_fastjet.py (seed=0) so the
timings line up with the flashjet GPU numbers measured in the b_hive env.
Times the classic per-event PseudoJet/ClusterSequence path (what one would
actually call to cluster real events on CPU).
"""
import sys, time
import numpy as np
import fastjet

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
        out.append(np.stack([pt*np.cos(phi), pt*np.sin(phi), mt*np.sinh(y), mt*np.cosh(y)], 1))
    return out


def timeit(fn, iters):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    print(f"fastjet {fastjet.__version__} | numpy {np.__version__}\n")
    jd = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    print(f"{'B':>5} {'N':>6} {'particles':>10} {'ms/batch':>10} {'us/event':>10} {'Mpart/s':>9}")
    for B, N, iters in [(1024, 32, 5), (1024, 64, 5), (1024, 128, 5),
                        (512, 512, 3), (256, 2048, 2), (128, 6000, 1)]:
        events = gen_events(B, N)
        ntot = sum(len(e) for e in events)

        def run():
            for ev in events:
                pjs = [fastjet.PseudoJet(*map(float, row)) for row in ev]
                fastjet.ClusterSequence(pjs, jd).inclusive_jets()

        t = timeit(run, iters)
        print(f"{B:>5} {N:>6} {ntot:>10} {t*1e3:>10.2f} {t/B*1e6:>10.1f} {ntot/t/1e6:>9.3f}")


if __name__ == "__main__":
    main()
