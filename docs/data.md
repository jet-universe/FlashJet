# Ragged events and GPU batches

Install the `data` extra for PyTorch, Awkward, and Parquet support.
`flashjet.data.collate` accepts a list of `(n_i, 4)` NumPy arrays or an Awkward
array with `px`, `py`, `pz`, and `E` fields (`e` and `energy` are also accepted).

```python
import numpy as np
from flashjet.data import collate

events = [np.array([[10., 0., 0., 10.]]),
          np.array([[2., 0., 0., 2.], [-5., 0., 0., 5.]])]
p4, mask = collate(events, n_max=8, truncate="error")
```

Collation produces float32 tensors. By default it pads to the longest event.
With `n_max`, the default `truncate="pt"` keeps the particles with highest
transverse momentum. `"first"` keeps the first particles; `"error"` rejects
oversized events. Truncation changes the physics input, so choose it explicitly.
The optional `out=(p4_buffer, mask_buffer)` reuses preallocated storage.

```python
import awkward as ak
import flashjet
from flashjet.data import to_gpu_batches, gpu_batch_ready

events = ak.from_parquet("events.parquet")
for batch in to_gpu_batches(events, batch_size=512, truncate="error"):
    p4, mask = gpu_batch_ready(batch)
    out = flashjet.cluster(p4, mask, R=0.4)
    jets = out.jets_p4(p4)
    # Consume this batch here.
```

`to_gpu_batches()` uses pinned host memory and asynchronous copies.
Always call `gpu_batch_ready()` before consuming a yielded batch: it handles
stream synchronization and tensor lifetime. The input dataset is flattened in
memory; this is not a streaming Parquet reader. Choose batch size and padded
width to fit host and device memory.
