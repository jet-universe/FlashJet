# Troubleshooting

## Triton backend unavailable

Check `torch.cuda.is_available()`, the input device, and whether Triton imports.
Use Linux x86_64 with a supported NVIDIA GPU and driver. The register kernel
accepts at most 128 padded slots; the large kernel accepts at most 16384.
A macOS install cannot provide a CUDA device.

## CPU install works but clustering is slow

Check `flashjet._native.HAS_NATIVE`. If false, the optional compiler step or
shared-library load failed. Reinstall from a wheel, or build from source with a
C++17 compiler. `HAS_OPENMP=False` means the native kernel is single-threaded;
it does not mean the kernel is missing.

## Invalid input or unexpected jets

Check the `(px, py, pz, E)` order, energy and momentum units, physical energies,
mask shape and device, and padded width. Non-finite masked-in values raise by
default. `R` must be positive. Padding must be masked out.

## Results differ slightly

Compare particle assignments separately from summed momenta. Precision,
near-ties, autotuning, and CUDA summation order can each matter. See
[Backends](backends.md) before expecting bitwise agreement with another setup.

## GPU memory is exhausted

Reduce batch size or padding. Do not use the cubic torch fallback for large
events. If you truncate inputs with `n_max`, record that choice in your analysis.

## A test skips

The CPU suite cannot run CUDA kernels. A green CPU run with GPU skips does not
validate Triton. Use the manually dispatched GPU workflow on a registered CUDA
runner, and check that its preflight confirms both CUDA and Triton.
