#!/usr/bin/env python
"""Benchmark triton vs torch backends on GPU.

Run on a CUDA machine:  python scripts/bench_gpu.py [B] [N]
"""

import sys
import time

import torch

import flashjet
from flashjet.torch_backend import cluster_batch_torch
from flashjet.triton_backend import HAS_TRITON, cluster_batch_triton
from flashjet.triton_large import MAX_LARGE_N, cluster_batch_triton_large
from flashjet.tune import _spin


def make_batch(B, N, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    pt = torch.rand(B, N, generator=g) * 80 + 0.5
    y = torch.randn(B, N, generator=g)
    phi = torch.rand(B, N, generator=g) * 6.283185
    m = torch.rand(B, N, generator=g)
    mt = (m**2 + pt**2).sqrt()
    p4 = torch.stack([pt * phi.cos(), pt * phi.sin(), mt * torch.sinh(y), mt * torch.cosh(y)], -1)
    n = torch.randint(N // 2, N + 1, (B,), generator=g)
    mask = torch.arange(N).expand(B, N) < n[:, None]
    return p4.to(device), mask.to(device)


def bench(fn, *args, iters=20):
    fn(*args)  # warmup / compile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    p4, mask = make_batch(B, N, "cuda")
    p4 = p4.float()
    _spin(p4.device)  # ramp clocks before sub-second timings

    t_torch = bench(cluster_batch_torch, p4, mask, 0.4, -1.0)
    print(f"torch  backend: {t_torch*1e3:8.2f} ms / batch of {B} events (N={N})")
    if HAS_TRITON and N <= 128:
        t_triton = bench(cluster_batch_triton, p4, mask, 0.4, -1.0)
        print(f"triton backend: {t_triton*1e3:8.2f} ms / batch  ({t_torch/t_triton:.1f}x vs torch)")
    else:
        print("triton backend unavailable (no triton or N > 128)")
    if HAS_TRITON and N <= MAX_LARGE_N:
        t_large = bench(cluster_batch_triton_large, p4, mask, 0.4, -1.0)
        print(f"triton-large  : {t_large*1e3:8.2f} ms / batch  ({t_torch/t_large:.1f}x vs torch)")
        t_api = bench(lambda: flashjet.cluster(p4, mask, R=0.4, algorithm="antikt", backend="triton-large"))
        print(f"triton-large via api (incl. history decode): {t_api*1e3:8.2f} ms / batch")
    else:
        print("triton-large backend unavailable (no triton or N > 16384)")


if __name__ == "__main__":
    main()
