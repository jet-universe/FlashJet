# Quickstart

## One NumPy event

```python
import numpy as np
import flashjet

p4 = np.array([[10., 0., 0., 10.], [2., 0., 0., 2.], [-5., 0., 0., 5.]])
seq = flashjet.cluster(p4, R=0.4)
print(seq.inclusive_jets())       # [[12, 0, 0, 12], [-5, 0, 0, 5]]
print(seq.jet_constituents())     # [[0, 1], [2]]
```

Each row is `(px, py, pz, E)` in one consistent unit. NumPy input always uses
the float64 reference implementation; `backend`, `mask`, and `decode` do not
select a different NumPy path. `inclusive_jets(ptmin)` uses the strict cut
`pt > ptmin` and returns jets in descending transverse momentum.

## A padded tensor batch

```python
import torch
import flashjet

p4 = torch.tensor([[[10., 0., 0., 10.], [2., 0., 0., 2.],
                    [-5., 0., 0., 5.], [0., 0., 0., 0.]]])
p4.requires_grad_()
mask = torch.tensor([[True, True, True, False]])
out = flashjet.cluster(p4, mask, R=0.4)
jets = out.jets_p4(p4)
sorted_jets, order = out.sort_jets_by_pt(jets)
sorted_jets[..., 3].sum().backward()
```

`p4` has shape `(batch, particles, 4)`. Use float32 or float64. `mask` has shape
`(batch, particles)`, is Boolean, and lives on the same device. Omit it only
when every slot is a real particle. Padding gets assignment `-1`.

Move both inputs to CUDA for GPU execution. `backend="auto"` chooses a backend
from the padded width and device; see [Backends](backends.md).

`out.n_jets` counts jets per event. Tensor jets start in beam-removal order,
not momentum order. If you sort the momenta, apply the returned `order` to any
per-jet features too. Requesting a smaller `n_jets_max` drops higher-index jets;
it is not a cut on transverse momentum.

## Algorithms and gradients

Choose `algorithm="antikt"`, `"kt"`, or `"cambridge"`. The aliases `"anti-kt"`,
`"ca"`, and `"cambridge-aachen"` also work. `p=` overrides the algorithm with a
generalized-kt exponent. Use a finite positive `R` and finite input momenta.

Clustering decisions run without gradients. `jets_p4()` sums the selected
particles with PyTorch operations, so derivatives flow through that fixed
assignment. This does not differentiate changes in jet membership.
