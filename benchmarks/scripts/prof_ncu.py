"""Minimal driver for Nsight Compute: one _cluster_large_kernel launch.
Triton's compiled kernel is already disk-cached from earlier runs, so the
single cluster call below is a real (non-JIT) launch ncu can profile.
"""
import sys, torch
sys.path.insert(0, "scripts")
from bench_gpu import make_batch
from flashjet.triton_large import cluster_batch_triton_large

B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
p4, mask = make_batch(B, N, "cuda"); p4 = p4.float()
cluster_batch_triton_large(p4, mask, 0.4, -1.0)
torch.cuda.synchronize()
