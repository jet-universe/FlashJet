"""Cluster the SAME real jets (saved by real_extract_flashjet.py) with FastJet,
both the classic per-event interface and the awkward-vectorized interface.
"""
import time, numpy as np, fastjet

events = np.load("/tmp/cgupta/real_jets.npy", allow_pickle=True)
B = len(events)
ntot = int(sum(len(e) for e in events))
R = 0.4
jd = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
print(f"loaded {B} real jets | total constituents {ntot}\n")


def timeit(fn, iters):
    r = fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter()-t0)/iters, r


# classic: per-event PseudoJet loop
def classic():
    njets = np.empty(B, dtype=np.int64)
    for i, ev in enumerate(events):
        pjs = [fastjet.PseudoJet(*map(float, row)) for row in ev]
        njets[i] = len(fastjet.ClusterSequence(pjs, jd).inclusive_jets())
    return njets

tc, njc = timeit(classic, 5)
print(f"FastJet classic  (CPU loop): {tc*1e3:8.2f} ms / {B} jets "
      f"({tc/B*1e6:7.2f} us/jet, {ntot/tc/1e6:.3f} Mpart/s)")

# awkward-vectorized: one ClusterSequence over all jets
try:
    import awkward as ak
    recs = ak.Array([[{"px": float(r[0]), "py": float(r[1]),
                       "pz": float(r[2]), "E": float(r[3])} for r in ev] for ev in events])

    def awk():
        cs = fastjet.ClusterSequence(recs, jd)
        return ak.num(cs.inclusive_jets(), axis=1)

    ta, nja = timeit(awk, 5)
    print(f"FastJet awkward  (CPU vect): {ta*1e3:8.2f} ms / {B} jets "
          f"({ta/B*1e6:7.2f} us/jet, {ntot/ta/1e6:.3f} Mpart/s)")
except Exception as e:
    print("awkward interface failed:", e)

# agreement vs flashjet on the same events
try:
    njf = np.load("/tmp/cgupta/real_njets_flashjet.npy")
    print(f"\nn_jets agreement flashjet vs FastJet-classic: "
          f"{int((njf == njc).sum())}/{B} ({100*(njf==njc).mean():.2f}%)")
except FileNotFoundError:
    pass
