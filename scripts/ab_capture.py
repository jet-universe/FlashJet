#!/usr/bin/env python
"""A/B bitwise fixture tool for the triton-large kernel.

capture: cluster fixed seeded batches and save every output tensor to disk.
compare: torch.equal two captured tags, tensor by tensor (exit 1 on mismatch).

Usage:
    python scripts/ab_capture.py capture <tag>
    python scripts/ab_capture.py compare <tagA> <tagB>
"""

import math
import os
import sys

import torch

from flashjet.triton_large import cluster_batch_triton_large

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".ab_fixtures")
SHAPES = [(256, 128), (256, 512), (64, 1024), (16, 4096)]
CONFIGS = {
    "default": {},
    "b512w4t32x64": {"block": 512, "num_warps": 4, "tile": (32, 64)},
}
R, P = 0.4, -1.0  # anti-kt


def make_batch(B, N, seed, device):
    # same event model as scripts/bench_pipeline.py make_parquet:
    # pt~U(0.5,80), y~U(-3,3), phi~U(0,2pi), m~U(0,1), f64 math -> f32
    g = torch.Generator(device="cpu").manual_seed(seed)
    pt = torch.rand(B, N, generator=g, dtype=torch.float64) * 79.5 + 0.5
    y = torch.rand(B, N, generator=g, dtype=torch.float64) * 6.0 - 3.0
    phi = torch.rand(B, N, generator=g, dtype=torch.float64) * 2 * math.pi
    m = torch.rand(B, N, generator=g, dtype=torch.float64)
    mt = (m**2 + pt**2).sqrt()
    p4 = torch.stack([pt * phi.cos(), pt * phi.sin(), mt * torch.sinh(y), mt * torch.cosh(y)], -1)
    n = torch.randint(N // 2, N + 1, (B,), generator=g)
    mask = torch.arange(N).expand(B, N) < n[:, None]
    return p4.float().to(device), mask.to(device)


def fixture_path(tag, B, N, cfg_name):
    return os.path.join(ROOT, tag, f"B{B}_N{N}_{cfg_name}.pt")


def capture(tag):
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    for i, (B, N) in enumerate(SHAPES):
        p4, mask = make_batch(B, N, seed=1000 + i, device="cuda")
        for cfg_name, knobs in CONFIGS.items():
            out = cluster_batch_triton_large(p4, mask, R, P, **knobs)
            path = fixture_path(tag, B, N, cfg_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({k: v.cpu() for k, v in out.items()}, path)
            print(f"captured {path}")


def compare(tag_a, tag_b):
    bad = []
    for B, N in SHAPES:
        for cfg_name in CONFIGS:
            a = torch.load(fixture_path(tag_a, B, N, cfg_name), weights_only=True)
            b = torch.load(fixture_path(tag_b, B, N, cfg_name), weights_only=True)
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b or not torch.equal(a[key], b[key]):
                    bad.append(f"B{B}_N{N}_{cfg_name}:{key}")
    if bad:
        print(f"MISMATCH ({len(bad)} tensors):")
        for item in bad:
            print(f"  {item}")
        sys.exit(1)
    print(f"PASS: {tag_a} == {tag_b} bitwise ({len(SHAPES) * len(CONFIGS)} fixtures, all tensors equal)")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "capture":
        capture(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__.strip())


if __name__ == "__main__":
    main()
