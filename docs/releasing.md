# Git distribution and release checklist

FlashJet is distributed through [GitHub](https://github.com/jet-universe/FlashJet).
The [documentation site](https://jet-universe.github.io/FlashJet/) follows `main`.
PyPI publishing is deferred; no workflow uploads to PyPI.

## Install from Git

Follow the [installation guide](installation.md) to install the default branch.
For reproducible work, append `@<full-commit-sha>` to the Git URL. A tag can also
select a revision once that tag exists. The version in package metadata does not
by itself mean a tagged release has been published.

## CI coverage

Installation and distribution validation run in GitHub Actions:

| Workflow | Checks |
|---|---|
| [CPU tests](https://github.com/jet-universe/FlashJet/actions/workflows/cpu.yml) | Exact Git commit installation and CPU tests on Python 3.9–3.13, Ubuntu and macOS; a separate NumPy-only installation |
| [Distributions](https://github.com/jet-universe/FlashJet/actions/workflows/build.yml) | 15 wheels and one source archive; installed-artifact smoke tests; artifact upload/download checks; strict metadata validation |
| [Documentation](https://github.com/jet-universe/FlashJet/actions/workflows/docs.yml) | Sphinx build with warnings treated as errors; previews on PRs; Pages deployment from `main` |
| [GPU tests](https://github.com/jet-universe/FlashJet/actions/workflows/gpu.yml) | Manual CUDA validation on a self-hosted GPU runner |

Check the results for the exact commit you intend to use. A successful docs
build does not establish CPU or GPU correctness, and queued or skipped jobs are
not validation results. CPU-only runs skip CUDA tests.

## Download CI artifacts

Open a successful **Distributions** run for your commit and find its artifacts
at the bottom of the run page. Downloading artifacts through GitHub requires
signing in. Extract the downloaded ZIP before installing a wheel.

| Artifact | Contents |
|---|---|
| `wheels-ubuntu-22.04` | Linux x86_64 wheels |
| `wheels-macos-15-intel` | macOS Intel wheels |
| `wheels-macos-14` | macOS Apple Silicon wheels |
| `source-distribution` | Source archive |

Each wheel artifact contains CPython 3.9–3.13 builds. Choose the wheel matching
your interpreter and platform. Linux wheels use the manylinux_2_28 baseline.
Wheels include the native CPU kernel and fail their smoke test if it cannot
load. PyTorch and Triton are external dependencies, not separate FlashJet wheel
variants. Linux wheel repair bundles required shared-library dependencies.

The `verify-distributions` job downloads the build artifacts, requires 15 wheels
and one source archive, and checks their metadata with Twine. The `roundtrip-*`
artifacts are small workflow test fixtures, not installable packages. CI
artifacts expire according to GitHub's retention settings; tagged releases are
the place for reviewed release downloads.

## Create a tagged GitHub release

1. Review the changes, contributor credits, and changelog. The original source
   provenance is recorded in `snapshot-source.json`.
2. Confirm CPU tests, distribution checks, and docs pass for the exact release
   commit. Record GPU results separately before claiming CUDA validation.
3. Create and push a tag matching the package version, such as `v0.1.0`.
4. The distribution workflow builds artifacts, verifies them, and creates a
   **draft** GitHub release. Review its files and notes, then publish the draft
   on GitHub when ready.

The tag workflow does not run the CPU or GPU suites itself. There is no PyPI
publication step.

## GPU validation

The manual CUDA workflow requires a Linux x86_64 self-hosted runner labelled
`gpu`, with an NVIDIA driver suitable for the selected PyTorch builds. Keep the
runner current so it supports the Node.js runtime used by GitHub Actions.

The matrix targets PyTorch 2.6.0 / Triton 3.2.0 on Python 3.10 and PyTorch 2.7.1 /
Triton 3.3.1 on Python 3.12. These are configured test targets, not a claim that
GPU validation has passed. Dispatch the workflow for the intended revision and
inspect both matrix jobs. Dependabot does not update these paired version
literals; maintain them together after compatibility testing.

## Documentation deployment

Changes pushed to `main` build and deploy the Sphinx site to GitHub Pages.
Pull requests build a downloadable `documentation-preview` without deploying.
The repository's Pages build source is **GitHub Actions**.

`.readthedocs.yaml` is available for an optional Read the Docs setup using the
same strict build. No Read the Docs project is provisioned by this repository.

The workflows follow the
[GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [cibuildwheel documentation](https://cibuildwheel.pypa.io/en/stable/).
