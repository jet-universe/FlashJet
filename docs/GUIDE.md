# flashjet, in plain English

This guide uses simple language. You only need Python and a rough idea of what a particle collision is.

The [README](https://github.com/jet-universe/FlashJet/blob/main/README.md) is the short technical version. [Backends](backends.md) explains backend selection and numerical behavior.

---

## 1. The problem

A particle detector records a spray of particles. Each collision has a few hundred to a few thousand of them. Most of those particles came from a small number of original objects that later split apart. Physics is done on those original objects, not on the fragments.

So you have to group the fragments back together. Each group is called a **jet**. Deciding which fragments belong to which jet is called **jet clustering**.

The standard recipe is used by almost every LHC experiment. It works like this:

1. Look at every pair of particles. Give each pair a number that says how much those two look like they came from the same thing.
2. Also give each *single* particle a number that says this one is finished. Nothing left to merge it with.
3. Find the smallest number in the whole set.
4. If it was a pair, glue those two particles into one and go back to step 1. If it was a single particle, declare it a finished jet and remove it.
5. Repeat until nothing is left.

That is the whole algorithm. The popular variants only change the formula in step 1. flashjet implements the three standard ones:

| name | what it merges first | typical use |
|---|---|---|
| `"antikt"` (default) | hard particles pull in their soft neighbours | the standard choice for finding jets |
| `"kt"` | the softest, closest pairs | reconstructing the splitting history |
| `"cambridge"` (a.k.a. C/A) | purely the closest pairs, ignoring energy | grooming and substructure |

The `R` parameter is the size of a jet. It sets the pair-distance scale, not a strict maximum separation between all final constituents. `R=0.4` is the usual default.

The reference implementation is [FastJet](https://fastjet.fr). It is a C++ library and has been the standard for twenty years. flashjet is a re-implementation of the same algorithm that runs on a GPU. We check it against real FastJet in the test suite.

---

## 2. Why this library exists

FastJet is excellent. Keep using it for ordinary analysis work. flashjet exists for one case: **you want to cluster jets in the middle of training a neural network.**

If your training data lives on the GPU and you want jets, the usual route is this: copy everything back to the CPU, call FastJet one event at a time in a Python loop, then copy the answers back to the GPU. That round trip is often slower than the training step it is feeding.

flashjet keeps everything where it already is. You hand it a batch of events as a GPU tensor. You get jets back as a GPU tensor. The Triton clustering kernels keep particle data on the GPU. Validation and some result-shape operations can still synchronize with the host.

The second reason is that the answer is **differentiable**. The grouping itself is not. Grouping is a discrete choice: a particle is either in a jet or it is not. Once the grouping is fixed, the jet's energy and momentum are just a sum over its members. flashjet gives you that sum as a normal torch operation, so gradients flow through it.

---

## 3. Five minutes to your first jets

Install it:

```bash
pip install -e '.[torch,test]'
```

The install also compiles a small C++ file. If you have no C++ compiler, that is fine. It prints a note and uses a slower pure-Python path.

Now cluster something:

```python
import torch, flashjet

# A batch of 8 events, each with room for 64 particles.
# The last axis is (px, py, pz, E) -- momentum and energy.
p4 = torch.randn(8, 64, 4).abs()
p4[..., 3] = p4[..., :3].pow(2).sum(-1).sqrt() + 0.1   # make E >= |p|

# Real events have different numbers of particles, so every event is padded
# to the same length and a mask says which slots are real.
mask = torch.ones(8, 64, dtype=torch.bool)

out = flashjet.cluster(p4, mask, R=0.4, algorithm="antikt")

print(out.n_jets)          # how many jets each event ended up with
print(out.jet_idx.shape)   # (8, 64): which jet each particle went into

jets = out.jets_p4(p4)     # (8, J, 4): the four-momentum of each jet
jets, order = out.sort_jets_by_pt(jets)   # hardest jet first
```

Put `p4` and `mask` on a CUDA device and the same call runs on the GPU. You do not change anything else.

### The two things you get back

**`out.jet_idx`** — shape `(events, particles)`. For each particle, the number of the jet it ended up in. `-1` means the slot was padding, not a real particle. This answers "which jet is this particle in".

**`out.jets_p4(p4)`** — shape `(events, jets, 4)`. The total momentum and energy of each jet. This answers "what does the jet look like". It is differentiable. `jet_idx` is not.

There is also `out.hist_p1`, `hist_p2`, `hist_child` and `hist_d`. These record the full sequence of merges: what got glued to what, in what order, and at what distance. Most people never touch these directly. Everything in section 5 is computed from them.

### Single events, NumPy style

If you just have one event as a NumPy array and want something FastJet-shaped:

```python
import numpy as np, flashjet

event = np.random.rand(40, 4)                 # (n, 4)
event[:, 3] = np.linalg.norm(event[:, :3], axis=1) + 0.1
cs = flashjet.cluster(event, R=0.4, algorithm="antikt")

cs.inclusive_jets(ptmin=5.0)     # (J, 4), hardest first
cs.jet_constituents(ptmin=5.0)   # which input particles went into each of them
```

This path uses double precision. It is a readable reference implementation used to test the faster backends.

---

## 4. Which engine runs, and why you can ignore it

There are five implementations of the same algorithm in this repository. You normally do not pick one. `backend="auto"` chooses:

- **CUDA tensor with Triton installed?** A GPU kernel handles padded widths up to 16384. Otherwise the torch backend is used.
- **CPU tensor?** The C++ kernel handles it if available, with threads when OpenMP is enabled. Otherwise it falls back to NumPy.
- **NumPy array?** The plain double-precision reference implementation.

You can force one with `backend="triton"`, `"triton-large"`, `"cpu"` or `"torch"` if you are benchmarking or debugging. The reference implementation is not a `backend=` choice. It is what you get by passing a NumPy array. For ordinary use, leave this alone.

The tests compare faster backends with simpler implementations and compare the reference with FastJet. Exact histories are checked where the numerical contract allows it; floating-point ties can differ between backends. See [Backends](backends.md) for limits.

---

## 5. Reading the jet's internal structure

Once you know how a jet was built up, you can ask how it split. That is often more useful than the jet's total energy. Common questions are built in. None of them re-cluster anything. They just read the recorded merge history.

These only make physical sense for `kt` and `cambridge` clustering, which build jets in a meaningful order.

Substructure is usually done on one *fat* jet. That means a wide `R` that puts the whole decay into a single jet, which you then take apart again. So cluster with a large `R`, not 0.4:

```python
out = flashjet.cluster(p4, mask, R=1.0, algorithm="cambridge")

# Recorded merge distances; use algorithm="kt" for kt splitting scales.
out.splitting_scales()

# Undo merges towards 3 pieces per event, where the recorded tree allows it.
# `idx` has the same shape and meaning as out.jet_idx -- one subjet number
# per particle -- so you sum it up the same way jets_p4 does.
idx, n = out.exclusive_jets(n_jets=3)

# Soft drop / mMDT: throw away the soft, wide-angle junk at the edge of a jet
# and keep the hard core. The usual first step before measuring a jet's mass.
g = out.groomed_jets(p4, R=1.0, z_cut=0.1, beta=0.0)
g["groomed_p4"]   # the cleaned-up jets
g["tagged"]       # did the grooming find a genuine two-prong split?

# Lund plane coordinates -- a standard input representation for ML on jets.
out.lund_coordinates(p4, R=1.0)
```

`exclusive_jets` can only *undo* merges that actually happened. Ask for 3 pieces of an event that clustered into 9 separate jets and you get 9 back, not 3. There is nothing to take apart. That means your `R` is too small for what you are trying to do. It is not a bug.

---

## 6. Feeding it real data

Real events come ragged. Every event has a different number of particles. They usually arrive as a parquet file or an awkward array. Padding that in a Python loop is slow enough to cancel the point of the library, so there is a helper:

```python
import awkward as ak, flashjet
from flashjet.data import to_gpu_batches, gpu_batch_ready

events = ak.from_parquet("events.parquet")    # needs px, py, pz, E fields

for batch in to_gpu_batches(events, batch_size=512):
    p4, mask = gpu_batch_ready(batch)         # <-- do not skip this line
    out = flashjet.cluster(p4, mask, R=0.4, algorithm="antikt")
```

It pads with one vectorised operation and copies to the GPU in the background. While you are clustering batch *k* it is already preparing batch *k+1*.

`gpu_batch_ready(batch)` is required. The copy happens on a separate stream. That call waits for it to finish. If you touch the tensors without it, you will eventually get garbage.

---

## 7. How fast is it

These numbers come from the original development measurements. They were not rerun for this release snapshot and are not performance guarantees.

**On a GPU, for many small events at once, it is a large win.** The end-to-end pipeline (parquet file → GPU → jets, transfers included) runs about 11–18 µs per event depending on the card. That is roughly 7× a single CPU core running FastJet. That factor is the reason the library exists.

One catch: one GPU program handles one event, so the GPU only fills up if you give it enough events at once. With big events (thousands of particles) you want batches of at least ~100 before the numbers look good.

**On a CPU, it is competitive with FastJet, not far ahead.** Roughly 1.1–1.4× faster per core across the sizes measured. FastJet's own lazy strategy catches up at the largest event sizes. See the [benchmark records](performance.md) for the measured workloads. Comparing our multi-threaded mode against single-threaded FastJet would be unfair, since FastJet threads too.

Benchmarking rules, because they have bitten us: take medians over many runs, check the machine is idle first, and never quote a single timing. Single-shot numbers on the development box were wrong by enough to reverse conclusions.

---

## 8. Things that will trip you up

**You must pass a mask** unless every event really is full. Padding slots left unmasked become fake particles and produce fake jets.

**`E` has to be at least `|p|`.** Random tensors usually are not valid four-momenta. If in doubt, build `E` from the momentum.

**GPU results use single precision.** When two candidate merges are almost exactly tied, the GPU and the double-precision reference can pick different ones. The jets come out essentially the same, but not bit-for-bit identical. This is expected. It is not a bug. The tests are written to allow it.

**`jet_idx` is not differentiable.** Only `jets_p4()` is. Clustering runs under `no_grad` on purpose.

**A single `inf` or `nan` in your input can kill the CUDA context.** `cluster()` checks for this by default. In a hot loop with data you trust, `validate=False` skips the check and the GPU sync that comes with it.

**Jets are numbered in the order they finished**, which is not sorted by energy. Use `sort_jets_by_pt` if you want the hardest jet first.

---

## 9. Where to look next

| you want | go to |
|---|---|
| the technical summary and benchmark tables | [Performance and benchmarks](performance.md) |
| backend selection, limits, and numerical behavior | [Backends](backends.md) |
| worked examples of every guarantee | the `tests/` directory |
| the original algorithm | [FastJet](https://fastjet.fr) and its manual |

FlashJet is licensed under GPL-3.0-or-later. Contributions must be compatible with that license. FastJet-derived code retains its upstream attribution.
