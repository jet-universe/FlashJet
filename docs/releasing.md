# Git distribution and release checklist

The current distribution channel is GitHub. There is no PyPI upload workflow,
no publishing token, and no PyPI account setup required for this snapshot.
No remote changes are made until Sitian reviews the local checkout.

## Install from Git

After the repository is pushed:

```bash
python -m pip install 'flashjet @ git+https://github.com/jet-universe/FlashJet.git'
python -m pip install 'flashjet[triton,data] @ git+https://github.com/jet-universe/FlashJet.git'
```

For reproducible work, append `@<full-commit-sha>` to the Git URL. After a
reviewed tag exists, `@v0.1.0` can select it. An unpushed local snapshot cannot
be installed from the remote URL yet. Install with `python -m pip install .`
from this checkout during review.

## Review and first push

1. Review `snapshot-source.json`, package changes, contributor credits, and docs.
2. Review the CI configuration. Installation, CPU tests, source/wheel builds,
   and installed-wheel smoke tests run in GitHub Actions after the approved push.
   Record CUDA tests separately; they require the GPU runner.
3. Review `git diff --cached` after staging. Make the initial local commit.
4. Only after approval, push to `jet-universe/FlashJet`.
5. In repository Settings → Pages, select **GitHub Actions** as the build source.
   The docs workflow publishes main to `https://jet-universe.github.io/FlashJet/`.
   Pull requests build a downloadable preview without deploying.
6. Optionally import the repository into Read the Docs. `.readthedocs.yaml`
   uses the same strict Sphinx build; no Read the Docs project is provisioned here.

## GitHub release artifacts

The distribution workflow builds CPython 3.9–3.13 wheels for Linux x86_64,
macOS Intel, and macOS ARM64, and a source archive. Wheels include the optional
native CPU kernel and fail their smoke test if it cannot load. Triton and
PyTorch are external dependencies, not separate FlashJet wheel variants.
The manylinux repair step packages required shared-library dependencies.

Once changes are approved and CI passes, create a tag matching the package
version, such as `v0.1.0`. Pushing that tag builds artifacts and creates a
**draft** GitHub release. Review and publish the draft on GitHub when ready.
There is no PyPI publication. The release workflow does not run CPU/GPU suites
itself: confirm their results for the exact release commit before tagging.

The CUDA workflow is manual and requires a Linux x86_64 self-hosted runner
labelled `gpu`, with an NVIDIA driver suitable for the selected PyTorch builds.
It tests PyTorch 2.6.0 / Triton 3.2.0 on Python 3.10 and PyTorch 2.7.1 /
Triton 3.3.1 on Python 3.12. These are test targets, not results from this local
snapshot. Register a runner, dispatch each matrix job, and inspect results
before claiming GPU release validation. Dependabot does not update these
paired version literals; maintain them together after compatibility testing.

## Local artifacts

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

Builds create `dist/flashjet-0.1.0.tar.gz` and a wheel for the local interpreter
and platform. GitHub Actions builds the broader platform matrix. Generated
artifacts and documentation output are ignored by Git.

The automated publishing patterns follow the
[GitHub Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [cibuildwheel documentation](https://cibuildwheel.pypa.io/en/stable/).
