# Serving FlashJet

Clustering is not inference, but a [Triton Inference
Server](https://github.com/triton-inference-server/server) is a practical way
to give a GPU to processes that cannot use one themselves: a reconstruction
framework that hands a module one event at a time, or jobs spread over a batch
farm. The server also supplies what FlashJet needs and a single caller cannot --
**requests that arrive together are clustered in one kernel launch**.

## The clustering entry point

`flashjet.serving.cluster_events` is independent of Triton: events in, a
particle → jet map per event out. It uses the Triton kernels when torch and a
GPU are available, FlashJet's C++ kernel when they are not, and the NumPy
reference as a last resort.

```python
from flashjet.serving import cluster_events

results = cluster_events(events, R=0.4, p=-1.0)   # events: list of (n_i, 4) arrays
for jet_idx, n_jets in results:
    ...                                            # jet_idx: (n_i,) int32, beam-merge order
```

`flashjet.jet_idx_from_history_np` decodes a merge history into that map in
plain NumPy, for callers without torch.

## A model repository

```bash
python -m flashjet.serving.model_repository /path/to/models        # writes models/flashjet/
tritonserver --model-repository=/path/to/models
```

The model takes one event per request:

| | name | type | shape |
|---|---|---|---|
| input | `p4` | FP64 | `[-1, 4]`, columns (px, py, pz, E) |
| input | `algo` | FP64 | `[2]`, (R, p) |
| output | `jet_idx` | INT32 | `[-1]`, jet of each particle |
| output | `n_jets` | INT32 | `[1]` |

`dynamic_batching` is on by default: it is what lets the server merge requests
from many clients into one kernel launch. The `backend` parameter in
`config.pbtxt` (`auto`, `gpu`, `native`, `numpy`) is checked when the model
loads, so a misconfigured server fails at start-up rather than on the first
request.

FlashJet must be importable **inside** the server. Installing it there is the
simple route; `PYTHONPATH` to a checkout also works.

## Notes from deploying this

Measured with `tritonserver` 26.04 (the `fastml/triton-torchgeo` image) and CMS
reconstruction as the client:

* **Batching is the whole point.** Whole-event clustering of ~320-particle
  events went from 125 events/s with one client stream to 1745 with sixteen,
  because only then does the server have a batch to work with. A single client
  sending one event at a time will not beat a CPU.
* **Cap the multiplicity.** `cluster_events` refuses events above
  `MAX_PARTICLES` (16384): beyond the large-N kernel's limit `flashjet.cluster`
  falls back to the O(N³) torch backend, which would take a whole padded batch
  down with it.
* **The C++ kernel needs no torch.** `_cpu_kernel.cpp` builds with a plain
  `g++ -O3 -std=c++17 -shared -fPIC`, which is what makes the CPU path usable in
  a server image that has only NumPy.
* **Watch the image's own libtorch.** Images that ship the Triton PyTorch
  backend may preload libraries linked against it (`torch_geometric` in the
  image above) and put that libtorch first on `LD_LIBRARY_PATH`. A pip-installed
  torch in the Python backend then fails with undefined symbols. Overriding
  `LD_PRELOAD` and `LD_LIBRARY_PATH` for the server process fixes it.
* **Versions:** torch 2.14 with triton 3.8 runs the kernels unchanged.

## Precision

The GPU kernels are float32 while the NumPy and C++ paths are double precision,
so near-degenerate merges can be ordered differently between backends; see
[backends](backends.md). Jets are returned as a particle → jet map, and the
caller sums the four-momenta, so those sums carry the caller's precision.
