# Installation

The base package needs Python 3.9 or later and NumPy. Release CI targets CPython
3.9–3.13; newer interpreters are not yet part of that matrix.

## Install from Git

```bash
python -m pip install 'flashjet @ git+https://github.com/jet-universe/FlashJet.git'
python -m pip install 'flashjet[torch] @ git+https://github.com/jet-universe/FlashJet.git'
python -m pip install 'flashjet[triton,data] @ git+https://github.com/jet-universe/FlashJet.git'
```

Choose the command for your workload. These commands install the current default
branch. For reproducible work, append `@<full-commit-sha>` to the Git URL,
replacing the placeholder with the revision you want. PyPI publishing is deferred.

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

## Wheels from CI

The [Distributions workflow](https://github.com/jet-universe/FlashJet/actions/workflows/build.yml)
builds wheels for CPython 3.9–3.13 on Linux x86_64 and macOS Intel/Apple Silicon.
Choose a successful run for your intended commit, download its platform artifact,
and extract the ZIP. Install the wheel matching your Python version and platform:

```bash
python -m pip install '/path/to/flashjet-<version>-<python>-<abi>-<platform>.whl'
```

Replace the example path with the extracted wheel's filename. Wheels contain the
native CPU kernel; PyTorch and Triton remain separate dependencies. See the
[artifact guide](releasing.md#download-ci-artifacts) for artifact names and checks.

## Install from a checkout

From the repository root, use `python -m pip install .`, or
`python -m pip install '.[torch]'` for tensor batches. Use `-e` for an editable
development install.

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
