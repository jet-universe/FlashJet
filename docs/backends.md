# Backends and numerical behavior

| Backend | Input | Padded width | Behavior |
|---|---|---|---|
| NumPy reference | One `(n, 4)` array | No fixed cap | float64 reference |
| `cpu` | CPU tensor | No fixed cap | C++ kernel, or NumPy fallback |
| `torch` | CPU/CUDA tensor | No fixed cap | Cubic work; use small events |
| `triton` | CUDA tensor | Up to 128 | Dense register kernel |
| `triton-large` | CUDA tensor | Up to 16384 | Nearest-neighbor kernel with scratch memory |

For tensor input, `auto` selects `triton` for CUDA widths up to 16,
`triton-large` for CUDA widths 17–16384, and `cpu` for CPU tensors. Without
Triton, or above the large-kernel limit, CUDA falls back to `torch`. An explicit
Triton request raises an error if unavailable. The width includes padding.
The torch path warns above 512 slots and can consume substantial memory.

Distances use rapidity and wrapped azimuth, with E-scheme recombination:

```text
d_ij = min(pt_i^(2p), pt_j^(2p)) * (delta_y^2 + delta_phi^2) / R^2
d_iB = pt_i^(2p)
```

The smallest distance determines each pair merge or beam removal. `R` controls
the pair-distance scale; it is not a guarantee that every pair of final
constituents is within `R` of one another.

## Precision and reproducibility

Inputs should describe physical four-vectors. Validation rejects non-finite
values in real slots, but does not enforce every physical constraint.
`validate=False` removes the finite check for trusted inputs and avoids that
check's device synchronization.

Near equal distances can produce different merge orders across floating-point
precisions or GPU launch settings. Tests compare backends and FastJet, but do
not prove identical histories for every possible input. Keep dtype, hardware,
launch settings, and software versions fixed when studying reproducibility.

CUDA `scatter_add` in `jets_p4()` can change the last few bits of summed momenta.
Use `torch.use_deterministic_algorithms(True)` when you require stable sums.
The default `jets_p4()` shape also reads the jet count on the host; pass a fixed
`n_jets_max` when avoiding that synchronization matters.

## History layout

`hist_p1`, `hist_p2`, `hist_child`, and `hist_d` have shape `(B, N)`.
Initial pseudojet IDs count real particles in mask order, not padded slot order.
Pair merges create new IDs. `-1` denotes a beam removal; `-2` marks padding in
integer history arrays. `hist_d` stores each chosen distance.

`decode=False` skips particle assignment only for `triton-large`. It leaves
`jet_idx=None`, so `jets_p4()` cannot run. History-based splitting scales remain
available. Other backends ignore this flag and return assignments.
