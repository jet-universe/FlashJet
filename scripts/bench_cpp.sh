#!/usr/bin/env bash
# Build and run the pure-C++ flashjet vs FastJet benchmark.
#
# Both sides are compiled C++ on identical events, so this is the honest
# comparison: the python `fastjet` bindings add per-event overhead that is not
# FastJet's fault and would flatter us.
#
# Usage: scripts/bench_cpp.sh [threads]
set -euo pipefail
cd "$(dirname "$0")/.."

THREADS="${1:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"

# headers + libs ship inside the pip-installed fastjet package
FJ=$(python -c "import fastjet, os; print(os.path.dirname(fastjet.__file__))")
LIBDIR=$([ -d "$FJ/lib64" ] && echo "$FJ/lib64" || echo "$FJ/lib")

OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
${CXX:-g++} -O3 -std=c++17 -ffp-contract=off -funroll-loops -fopenmp \
  -I"$FJ/include" -o "$OUT/bench" \
  bench/bench_cpp_vs_fastjet.cpp src/flashjet/_cpu_kernel.cpp \
  -L"$LIBDIR" -lfastjet -Wl,-rpath,"$LIBDIR"

ROUNDS="${ROUNDS:-21}"
echo "anti-kt R=0.4.  Each point interleaves both sides for $ROUNDS rounds and reports"
echo "median [16th, 84th percentile]; the ratio is formed within each round."
echo
echo "1 thread:"
for a in "1024 20" "512 32" "256 64" "256 128" "64 256" "32 400" "32 700" "16 1000" "8 3000" "4 6000"; do
  "$OUT/bench" $a "$ROUNDS" 0.4 1 2>/dev/null | grep '^N=' || true
done
echo
echo "$THREADS threads, both sides (batch sizes raised so the pool balances):"
for a in "2048 20" "512 64" "512 128" "256 256" "64 1000" "32 3000" "16 6000"; do
  "$OUT/bench" $a "$ROUNDS" 0.4 "$THREADS" 2>/dev/null | grep '^N=' || true
done
