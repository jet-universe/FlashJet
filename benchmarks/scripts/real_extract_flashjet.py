"""Extract real ttbar jet constituents ONCE, save them, and time flashjet.
The saved file is consumed by real_fastjet.py so both clusterers see the
identical events.
"""
import sys, time, numpy as np, torch, uproot, flashjet

FILE = "/eos/user/c/cgupta/training-samples/hlt/out_TT_11.root"
TREE = "DeepJetNTupler/DeepJetvars"
NJETS = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
SAVE = "/tmp/cgupta/real_jets.npy"
R, P = 0.4, -1.0

t = uproot.open(FILE)[TREE]
arr = t.arrays(["jet_pfcand_pt", "jet_pfcand_eta", "jet_pfcand_phi", "jet_pfcand_mass"],
               entry_stop=NJETS, library="np")
pt, eta, phi, mass = (arr[f"jet_pfcand_{k}"] for k in ["pt", "eta", "phi", "mass"])
ncon = np.array([len(x) for x in pt]); keep = ncon > 0
pt, eta, phi, mass, ncon = pt[keep], eta[keep], phi[keep], mass[keep], ncon[keep]
B, N = len(pt), int(ncon.max())

events = []
for i in range(B):
    px = pt[i]*np.cos(phi[i]); py = pt[i]*np.sin(phi[i])
    pz = pt[i]*np.sinh(eta[i]); E = np.sqrt(px*px+py*py+pz*pz+mass[i]**2)
    events.append(np.stack([px, py, pz, E], 1).astype("float64"))
np.save(SAVE, np.array(events, dtype=object), allow_pickle=True)
print(f"saved {B} real jets -> {SAVE} | constituents/jet mean={ncon.mean():.1f} "
      f"max={N} total={ncon.sum()}")

p4 = torch.zeros(B, N, 4); mask = torch.zeros(B, N, dtype=torch.bool)
for i in range(B):
    p4[i, :ncon[i]] = torch.from_numpy(events[i].astype("float32"))
    mask[i, :ncon[i]] = True
p4c, maskc = p4.cuda(), mask.cuda(); torch.cuda.synchronize()
flashjet.cluster(p4c, maskc, R=R, p=P)  # warmup
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(10):
    out = flashjet.cluster(p4c, maskc, R=R, p=P)
torch.cuda.synchronize(); dt = (time.perf_counter()-t0)/10
np.save("/tmp/cgupta/real_njets_flashjet.npy", out.n_jets.cpu().numpy())
print(f"flashjet GPU (auto): {dt*1e3:.2f} ms / {B} jets  "
      f"({dt/B*1e6:.2f} us/jet, {ncon.sum()/dt/1e6:.2f} Mpart/s)")
