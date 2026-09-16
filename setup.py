"""Build the optional C++ clustering kernel.

It is a plain shared library loaded with ctypes (see _native.py), not a
CPython extension module -- setuptools is only used here to get a portable
compiler invocation and to drop the .so next to the package.  The build is
optional: if it fails, flashjet still works through the NumPy backend.
"""

import os
import sys
import tempfile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# -ffp-contract=off: no FMA fusion, so the C++ arithmetic matches the
# NumPy mirror bit for bit and the merge order cannot drift
_UNIX_FLAGS = ["-O3", "-std=c++17", "-funroll-loops", "-ffp-contract=off"]
_MSVC_FLAGS = ["/O2", "/std:c++17", "/fp:precise"]

# The probe must include <omp.h> and link, not just compile: Apple clang
# accepts `-Xpreprocessor -fopenmp` on a trivial translation unit and defines
# _OPENMP for it, so a header-less probe reports OpenMP available and the real
# build then dies on the missing libomp.
_OMP_PROBE = """
#include <omp.h>
int main() {
  int n = 0;
#pragma omp parallel
  { n = omp_get_max_threads(); }
  return n > 0 ? 0 : 1;
}
"""


class BuildExt(build_ext):
    """Add OpenMP when the toolchain really has it; never fail the install."""

    def build_extensions(self):
        ct = self.compiler.compiler_type
        base = _MSVC_FLAGS if ct == "msvc" else list(_UNIX_FLAGS)
        cargs, largs = self._openmp_flags(ct)
        if not cargs:
            print("flashjet: no working OpenMP; the C++ kernel will be single-threaded")
        for ext in self.extensions:
            ext.extra_compile_args = base + cargs
            ext.extra_link_args = list(largs)
        try:
            super().build_extensions()
            return
        except Exception as exc:  # noqa: BLE001
            if not cargs:
                print(f"flashjet: C++ kernel not built ({exc}); using the NumPy CPU backend")
                return
            # A threaded kernel is a nice-to-have; a kernel is not.  Threads are
            # a pure performance knob here (no shared mutable state), so dropping
            # OpenMP changes nothing but speed.
            print(f"flashjet: OpenMP build failed ({exc}); retrying single-threaded")
        for ext in self.extensions:
            ext.extra_compile_args = list(base)
            ext.extra_link_args = []
        try:
            super().build_extensions()
        except Exception as exc:  # noqa: BLE001
            print(f"flashjet: C++ kernel not built ({exc}); using the NumPy CPU backend")

    def _openmp_flags(self, ct):
        """Probe OpenMP, avoiding duplicate PyTorch runtimes on macOS."""
        mode = os.environ.get("FLASHJET_OPENMP", "auto")
        if mode == "0" or (sys.platform == "darwin" and mode != "1"):
            # PyTorch wheels bundle libomp. Linking Homebrew's second copy can
            # abort the process on the first parallel region (OMP Error #15).
            return [], []
        if ct == "msvc":
            return (["/openmp"], []) if self._links(["/openmp"], []) else ([], [])
        candidates = [(["-fopenmp"], ["-fopenmp"])]
        if sys.platform == "darwin":
            # Apple clang has no -fopenmp; it needs the runtime passed by hand,
            # and libomp is keg-only under Homebrew so it is not on the default
            # include path either.
            for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp", "/opt/local"):
                if os.path.isdir(prefix):
                    candidates.append((
                        ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
                        [f"-L{prefix}/lib", "-lomp"],
                    ))
            candidates.append((["-Xpreprocessor", "-fopenmp"], ["-lomp"]))
        for cargs, largs in candidates:
            if self._links(cargs, largs):
                return cargs, largs
        return [], []

    def _links(self, cargs, largs):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "probe.cpp")
            with open(src, "w") as fh:
                fh.write(_OMP_PROBE)
            try:
                objs = self.compiler.compile([src], output_dir=tmp, extra_postargs=list(cargs))
                self.compiler.link_executable(
                    objs, "probe", output_dir=tmp, extra_postargs=list(largs),
                    target_lang="c++",
                )
            except Exception:  # noqa: BLE001
                return False
            return True


setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=[] if os.environ.get("FLASHJET_NO_NATIVE") == "1" else [
        Extension(
            "flashjet._flashjet_cpu",
            sources=["src/flashjet/_cpu_kernel.cpp"],
            language="c++",
            optional=True,
        )
    ],
)
