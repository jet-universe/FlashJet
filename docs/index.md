# FlashJet

FlashJet groups particles into jets. It accepts a NumPy event or a padded batch
of PyTorch tensors. CUDA tensors can stay on the GPU during clustering.

Use it when a training loop needs to cluster jets repeatedly. The particle
assignment is discrete. Summing each jet's four-momentum remains differentiable
with respect to its assigned particles.

This is the documentation for the **0.1.0 release candidate**. The Git snapshot and
site are prepared for publication; links become live after review and push.

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
