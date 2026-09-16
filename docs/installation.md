# Installation

The base package needs Python 3.9 or later and NumPy. Release CI targets CPython
3.9–3.13; newer interpreters are not yet part of that matrix.

Before the first release, install from the repository root:

```bash
python -m pip install .
python -m pip install '.[torch]'        # padded tensor batches
python -m pip install '.[triton,data]'  # Linux x86_64 NVIDIA GPU and data I/O
```

Once the reviewed snapshot is pushed, install from Git:

```bash
python -m pip install 'flashjet[torch] @ git+https://github.com/jet-universe/FlashJet.git'
```

Append `@<full-commit-sha>` to pin a revision. PyPI publishing is deferred.
The extras are:

| Extra | Adds | Use |
|---|---|---|
| `torch` | PyTorch | CPU or CUDA tensor batches |
| `triton` | PyTorch and Linux x86_64 Triton | CUDA clustering |
| `data` | PyTorch, Awkward, PyArrow | Ragged events and Parquet files |
| `test` | pytest, FastJet, Awkward | Validation; also install `torch` for the full suite |

Install the CUDA-enabled PyTorch build suitable for your machine using the
[PyTorch instructions](https://pytorch.org/get-started/locally/) before adding
FlashJet. Keep the Triton version selected by that PyTorch release. Installing
an extra cannot provide an NVIDIA driver or GPU.

The Triton dependency marker only installs Triton on Linux x86_64. On macOS,
`[triton]` installs PyTorch but does not enable CUDA. Windows, ROCm, MPS, and
Linux ARM GPU execution are not validated release targets.

## Native CPU kernel

Release wheels must contain a loadable C++ kernel. Source installs try to build
it with a C++17 compiler, then fall back to NumPy if compilation fails. OpenMP
is optional; without it the native kernel runs on one thread.

```python
from flashjet import _native
print(_native.HAS_NATIVE, _native.HAS_OPENMP)
```

To build without the native kernel:

```bash
FLASHJET_NO_NATIVE=1 python -m pip install --no-cache-dir .
```

Use a clean build directory when switching native build modes. macOS builds
use a single-threaded C++ kernel by default: PyTorch bundles an OpenMP runtime,
and loading a second Homebrew runtime can abort the process. `FLASHJET_OPENMP=1`
opts into a probe when you control that runtime setup. `FLASHJET_OPENMP=0`
disables OpenMP on any platform. Linux builds probe for it by default.
