# FlashJet

Batched jet clustering for physics analysis and PyTorch training loops.
FlashJet implements anti-kt, kt, and Cambridge/Aachen with NumPy, an optional
C++ CPU kernel, PyTorch, and Triton GPU kernels.

[Documentation](https://jet-universe.github.io/FlashJet/) ·
[GitHub releases](https://github.com/jet-universe/FlashJet/releases) ·
[Source](https://github.com/jet-universe/FlashJet) ·
[Authors](AUTHORS.md)

**Release candidate:** publication is pending review. The documentation and
GitHub release links above are the intended destinations, not claims of a live release.

## Install

From this checkout, before publication:

```bash
python -m pip install '.[torch]'
# Linux x86_64 with a compatible NVIDIA driver and CUDA-enabled PyTorch:
python -m pip install '.[triton,data]'
```

After the reviewed snapshot is pushed, install directly from Git:

```bash
python -m pip install 'flashjet @ git+https://github.com/jet-universe/FlashJet.git'
python -m pip install 'flashjet[triton,data] @ git+https://github.com/jet-universe/FlashJet.git'
```

Append `@<full-commit-sha>` to the Git URL to pin an exact revision. PyTorch and
Triton remain optional dependencies. There is no PyPI publication in this setup.

## Cluster a batch

```python
import torch
import flashjet

p4 = torch.tensor([[[10., 0., 0., 10.],
                    [2., 0., 0., 2.],
                    [-5., 0., 0., 5.]]], requires_grad=True)
mask = torch.ones(p4.shape[:2], dtype=torch.bool)
out = flashjet.cluster(p4, mask, R=0.4, algorithm="antikt")
jets = out.jets_p4(p4)
jets, order = out.sort_jets_by_pt(jets)
jets[..., 3].sum().backward()
```

The last axis is `(px, py, pz, E)`. Use `p4.to("cuda")` and a mask on the same
device for GPU clustering. Gradients pass through the sum of constituent
momenta, not through the clustering decisions.

For one event, `flashjet.cluster(numpy_array)` returns a `ClusterSequenceRef`.
Call `inclusive_jets()` for momenta or `jet_constituents()` for particle indices.

Read the [quickstart](docs/quickstart.md), [plain-English guide](docs/GUIDE.md),
[backend limits](docs/backends.md), and [release checklist](docs/releasing.md).

## Develop

```bash
python -m pip install -e '.[torch,test,data]'
pytest -q
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

CUDA tests skip when no GPU is available. Release automation builds wheels
for CPython 3.9–3.13 on Linux x86_64 and macOS Intel/Apple Silicon, plus a source
archive. See the release checklist for what has actually been tested locally.

## License and credit

GPL-3.0-or-later; see [LICENSE](LICENSE). The clustering algorithm derives from
[FastJet](https://fastjet.fr). Cite its papers in physics publications.
Authors: Sitian Qian, Chirayu Gupta (@Chirayu18), and Alexandre De Moor
(@AlexDeMoor), with AI co-authors OpenAI Codex and Anthropic Claude. See [AUTHORS.md](AUTHORS.md).
