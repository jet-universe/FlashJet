# Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[torch,test,data]'
pytest -q
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
python -m pip install build twine
python -m build
python -m twine check dist/*
```

On Linux CPU-only machines, install PyTorch from its CPU wheel index first to
avoid downloading CUDA libraries. GPU tests skip without CUDA.

CI installs the exact checked-out Git commit with extras, tests the NumPy-only
install separately, and exercises built wheels and the source archive.

The tests compare the reference against FastJet, tensor backends against the
reference, and native histories against the NumPy CPU implementation. Additional
tests cover masks, tie handling, substructure, data collation, and decoding.
Wheel smoke tests run outside the checkout to verify the installed artifact.

Documentation uses Sphinx, MyST Markdown, and the Read the Docs theme. Public
API pages are generated from source docstrings. Keep examples runnable and
explain tensor shapes, units, defaults, and limits.

For prose, apply the rules in the local claudish-to-english repository's
`rewrite-md.sh`: short sentences, everyday words, unchanged technical facts,
links, paths, code blocks, and Markdown structure. Its language cleanup is a
hook, not the repository's unrelated release skill. The publication edit uses
those rules directly; it does not claim an external rewrite model ran.

Dependabot checks Python requirements and GitHub Actions weekly. Dependency
updates still need review and tests; no automatic merge is configured.
