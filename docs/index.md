# FlashJet

FlashJet groups particles into jets. It accepts a NumPy event or a padded batch
of PyTorch tensors. CUDA tensors can stay on the GPU during clustering.

Use it when a training loop needs to cluster jets repeatedly. The particle
assignment is discrete. Summing each jet's four-momentum remains differentiable
with respect to its assigned particles.

This site follows the published `main` branch. Install FlashJet from Git using
the [installation guide](installation.md). The package version is `0.1.0`; a
development snapshot is not a tagged release. PyPI publishing is deferred.

See [Git distribution and releases](releasing.md) for CI coverage, downloadable
artifacts, and the checks required before a release.

```{toctree}
:maxdepth: 2

installation
quickstart
GUIDE
backends
substructure
data
api
performance
troubleshooting
development
releasing
credits
```

[Source](https://github.com/jet-universe/FlashJet) ·
[GitHub releases](https://github.com/jet-universe/FlashJet/releases) ·
[Issues](https://github.com/jet-universe/FlashJet/issues)
