# Changelog

## 0.1.0 — development snapshot

First publication snapshot of FlastJetDemo. Includes the NumPy reference,
optional C++ CPU kernel, PyTorch and Triton backends, merge-history tools,
data batching, tests, and benchmark records.

The Git snapshot includes live GitHub Pages documentation, package builds, release workflows,
and dependency updates. It also makes the NumPy-only import independent of
PyTorch and adds `FLASHJET_NO_NATIVE=1` for compiler-free source builds.

macOS native builds default to single-threaded execution to avoid duplicate
OpenMP runtimes with PyTorch. `FLASHJET_OPENMP=1` opts into the compiler probe.

This publication uses GPL-3.0-or-later, electing a later version under the
original GPL-2.0-or-later terms. FastJet attribution is retained.
