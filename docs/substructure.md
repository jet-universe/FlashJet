# Substructure and grooming

The recorded merge tree supports later measurements without running clustering
again. Use kt for exclusive splitting scales and Cambridge/Aachen for the usual
angular declustering and grooming interpretation.

```python
out = flashjet.cluster(p4, mask, R=0.4, algorithm="cambridge")
lund = out.lund_coordinates(p4, R=0.4)
groomed = out.groomed_jets(p4, R=0.4, z_cut=0.1, beta=0.0)
```

Pass the same input momenta, radius, and mask used during clustering.
`cluster()` stores the mask on the output. Manually constructed outputs need
`mask=` for helpers that map padded slots to pseudojet IDs.

| Method | Result |
|---|---|
| `splitting_scales()` | `(B, J, S)` recorded distances in reverse merge order |
| `exclusive_jets(n_jets=3)` | `(assignment, count)` after undoing merges |
| `exclusive_jets(d_cut=...)` | Exclusive distance cut; pass only one cut mode |
| `lund_coordinates(p4, R)` | `(B, J, S, 6)` channels `(z, dR, kt, ln(1/dR), ln(kt), d)` |
| `groomed_jets(p4, R, ...)` | Dictionary containing groomed momenta and tag information |
| `mass_drop(p4, R, ...)` | Convenience wrapper with a mass cut and `z_cut=y_cut` |

The grooming dictionary contains `groomed_p4`, `tagged`, `z`, `dR`, `mu_split`,
and `n_drop`. Check `tagged` before interpreting a passing split.
The `mass_drop` wrapper uses this implementation's `z_cut=y_cut` convention;
do not assume it implements every convention of other taggers.

Per-jet features use the same unsorted jet order as `jets_p4()`. Gather them
with the same sorting indices if you sort the momenta. Split arrays are padded
with zeros. Their order follows the tree; it is not a general numerical sort.
The physical interpretation of a distance cut depends on the clustering
algorithm. Do not interpret anti-kt distances as kt splitting scales.

`exclusive_jets()` returns assignments, not four-momenta. You can use
`dataclasses.replace(out, jet_idx=assignment, n_jets=count).jets_p4(p4)` to sum
them without changing the original output.
