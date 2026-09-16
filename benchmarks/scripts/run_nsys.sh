#!/bin/bash
# Proper Nsight Systems profiling run for flashjet triton-large.
# Usage: bash run_nsys.sh [B] [N] [iters]
set -e
B=${1:-256}; N=${2:-2048}; ITERS=${3:-10}
NSYS=/opt/nvidia/nsight-systems/2025.6.3/bin/nsys
PY=/eos/home-c/cgupta/EPR_task/b-hive/micromamba/envs/b_hive/bin/python
REPO=/eos/home-c/cgupta/flashjet/FlastJetDemo
OUT=/tmp/cgupta/flashjet_prof_B${B}_N${N}
cd "$REPO"

echo "### capturing timeline -> ${OUT}.nsys-rep"
"$NSYS" profile --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true --force-overwrite=true \
    -o "$OUT" \
    "$PY" /tmp/cgupta/prof_flashjet.py "$B" "$N" "$ITERS"

echo "### GPU kernel summary"
"$NSYS" stats --report cuda_gpu_kern_sum --format table "${OUT}.nsys-rep" 2>/dev/null | head -30
echo "### GPU memory-op summary"
"$NSYS" stats --report cuda_gpu_mem_time_sum --format table "${OUT}.nsys-rep" 2>/dev/null | head -20
echo "### NVTX range summary (phase wall-time)"
"$NSYS" stats --report nvtx_sum --format table "${OUT}.nsys-rep" 2>/dev/null | head -20
