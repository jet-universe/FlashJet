"""Run against the installed wheel, outside the source directory, without torch."""
import sys
import numpy as np
import flashjet
from flashjet import _native

assert "torch" not in sys.modules, "base import must not require PyTorch"
assert _native.HAS_NATIVE, "release wheel must contain a loadable C++ kernel"
p4 = np.array([[10., 0., 0., 10.], [2., 0., 0., 2.], [-5., 0., 0., 5.]])
seq = flashjet.cluster(p4)
np.testing.assert_allclose(seq.inclusive_jets(), [[12., 0., 0., 12.], [-5., 0., 0., 5.]])
hp1, hp2, child, d = _native.cluster_native(
    p4[None], np.ones((1, 3), dtype=np.uint8), 0.4, -1., 1)
np.testing.assert_array_equal(hp1[0], [h.parent1 for h in seq.history])
np.testing.assert_array_equal(hp2[0], [h.parent2 for h in seq.history])
np.testing.assert_array_equal(child[0], [h.child for h in seq.history])
np.testing.assert_allclose(d[0], [h.d for h in seq.history])
print("Installed NumPy API and native kernel passed")
