"""Run flashjet on REAL CMS events: recluster the PF constituents of real
ttbar jets (DeepJet ntuple) on the GPU, and cross-check a sample against FastJet.

Reads jet_pfcand_{pt,eta,phi,mass} (jagged: one list of constituents per jet),
builds (B, N, 4) (px,py,pz,E) + mask, runs flashjet.cluster (anti-kt R=0.4).
"""
import sys, time, numpy as np, torch, uproot, flashjet

FILE = "/eos/user/c/cgupta/training-samples/hlt/out_TT_11.root"
TREE = "DeepJetNTupler/DeepJetvars"
NJETS = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
R, P = 0.4, -1.0

t = uproot.open(FILE)[TREE]
arr = t.arrays(["jet_pfcand_pt", "jet_pfcand_eta", "jet_pfcand_phi", "jet_pfcand_mass"],
               entry_stop=NJETS, library="np")
pt, eta, phi, mass = (arr[f"jet_pfcand_{k}"] for k in ["pt", "eta", "phi", "mass"])

ncon = np.array([len(x) for x in pt])
keep = ncon > 0
pt, eta, phi, mass, ncon = pt[keep], eta[keep], phi[keep], mass[keep], ncon[keep]
B = len(pt)
N = int(ncon.max())
print(f"real ttbar jets: {B} jets, constituents/jet min={ncon.min()} "
      f"max={N} mean={ncon.mean():.1f} total={ncon.sum()}")

p4 = torch.zeros(B, N, 4)
mask = torch.zeros(B, N, dtype=torch.bool)
for i in range(B):
    n = ncon[i]
    px = pt[i]*np.cos(phi[i]); py = pt[i]*np.sin(phi[i])
    pz = pt[i]*np.sinh(eta[i]); E = np.sqrt(px*px+py*py+pz*pz+mass[i]**2)
    p4[i, :n] = torch.from_numpy(np.stack([px, py, pz, E], 1).astype("float32"))
    mask[i, :n] = True

p4c, maskc = p4.cuda(), mask.cuda()
torch.cuda.synchronize()
# warmup (JIT)
flashjet.cluster(p4c, maskc, R=R, p=P)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    out = flashjet.cluster(p4c, maskc, R=R, p=P)
torch.cuda.synchronize()
dt = (time.perf_counter()-t0)/10
njet = out.n_jets.cpu().numpy()
print(f"flashjet GPU (auto backend): {dt*1e3:.2f} ms / {B} jets  "
      f"({dt/B*1e6:.1f} us/jet, {ncon.sum()/dt/1e6:.2f} Mpart/s)")
print(f"reclustered subjets/jet: mean={njet.mean():.2f} min={njet.min()} max={njet.max()}")

# cross-check a handful against FastJet-validated numpy reference (float64)
from flashjet.reference import cluster_event
nchk, agree = 20, 0
for i in range(nchk):
    n = ncon[i]
    ev = p4[i, :n].numpy().astype("float64")
    ref = cluster_event(ev, R=R, p=P)
    rj = len(ref.beam_jets)
    if rj == int(njet[i]):
        agree += 1
print(f"njets agreement vs float64 reference on {nchk} real jets: {agree}/{nchk}")
