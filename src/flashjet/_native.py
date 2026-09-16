"""ctypes binding for the compiled C++ clustering kernel (_cpu_kernel.cpp).

The kernel is a plain shared library with an `extern "C"` surface -- no
Python C-API, no pybind11 -- so it is loaded with ctypes and takes raw
pointers to numpy buffers.  If it was not built (source checkout without a
compiler, wheel-less install), HAS_NATIVE is False and cpu_backend falls back
to its NumPy implementation.
"""

import ctypes
import glob
import os

import numpy as np

_LIB = None
HAS_NATIVE = False
HAS_OPENMP = False


def _load():
    global _LIB, HAS_NATIVE, HAS_OPENMP
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    for pat in ("_flashjet_cpu*.so", "_flashjet_cpu*.dylib", "_flashjet_cpu*.pyd"):
        cands += sorted(glob.glob(os.path.join(here, pat)))
    for path in cands:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        i64p = ctypes.POINTER(ctypes.c_int64)
        for name, fptr in (("flashjet_cluster_f64", ctypes.POINTER(ctypes.c_double)),
                           ("flashjet_cluster_f32", ctypes.POINTER(ctypes.c_float))):
            fn = getattr(lib, name)
            fn.restype = None
            fn.argtypes = [fptr, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64,
                           ctypes.c_int64, ctypes.c_double, ctypes.c_double,
                           i64p, i64p, i64p, fptr, ctypes.c_int]
        lib.flashjet_has_openmp.restype = ctypes.c_int
        lib.flashjet_has_openmp.argtypes = []
        _LIB = lib
        HAS_NATIVE = True
        HAS_OPENMP = bool(lib.flashjet_has_openmp())
        return


_load()

_PTR = {np.dtype(np.float64): ctypes.POINTER(ctypes.c_double),
        np.dtype(np.float32): ctypes.POINTER(ctypes.c_float)}


def cluster_native(p4, mask, R, p, threads):
    """Cluster a padded (B, N, 4) numpy batch with the C++ kernel.

    Args:
        p4:   (B, N, 4) C-contiguous float32/float64, columns px, py, pz, E.
        mask: (B, N) C-contiguous uint8, nonzero for real particles.
    Returns (hist_p1, hist_p2, hist_child, hist_d) numpy arrays.
    """
    B, N, _ = p4.shape
    ft = p4.dtype
    hp1 = np.full((B, N), -2, dtype=np.int64)
    hp2 = np.full((B, N), -2, dtype=np.int64)
    hch = np.full((B, N), -2, dtype=np.int64)
    hd = np.zeros((B, N), dtype=ft)
    if B and N:
        fn = _LIB.flashjet_cluster_f64 if ft == np.float64 else _LIB.flashjet_cluster_f32
        cast = _PTR[ft]
        i64 = ctypes.POINTER(ctypes.c_int64)
        fn(p4.ctypes.data_as(cast),
           mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
           ctypes.c_int64(B), ctypes.c_int64(N), ctypes.c_double(R), ctypes.c_double(p),
           hp1.ctypes.data_as(i64), hp2.ctypes.data_as(i64), hch.ctypes.data_as(i64),
           hd.ctypes.data_as(cast), ctypes.c_int(threads))
    return hp1, hp2, hch, hd
