"""Nsight Systems profiling driver for the flashjet triton-large backend.

Run under nsys, e.g.:
  nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
       -o /tmp/cgupta/flashjet_prof python prof_flashjet.py 256 2048

NVTX ranges separate the phases so the timeline is readable:
  warmup (JIT, excluded) -> [kernel-only] cluster_batch_triton_large
                          -> [api+decode]  flashjet.cluster(backend=triton-large)
"""
import sys, torch, flashjet
sys.path.insert(0, "scripts")
from bench_gpu import make_batch
from flashjet.triton_large import cluster_batch_triton_large
from flashjet.tune import _spin

nvtx = torch.cuda.nvtx

B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 10

p4, mask = make_batch(B, N, "cuda")
p4 = p4.float()
_spin(p4.device)  # ramp clocks before timing

# warmup / JIT compile OUTSIDE any nvtx range so it doesn't pollute the timeline
cluster_batch_triton_large(p4, mask, 0.4, -1.0)
flashjet.cluster(p4, mask, R=0.4, algorithm="antikt", backend="triton-large")
torch.cuda.synchronize()

for i in range(ITERS):
    nvtx.range_push(f"kernel-only/iter{i}")
    cluster_batch_triton_large(p4, mask, 0.4, -1.0)
    torch.cuda.synchronize()
    nvtx.range_pop()

for i in range(ITERS):
    nvtx.range_push(f"api+decode/iter{i}")
    flashjet.cluster(p4, mask, R=0.4, algorithm="antikt", backend="triton-large")
    torch.cuda.synchronize()
    nvtx.range_pop()

print(f"profiled B={B} N={N} iters={ITERS}")
