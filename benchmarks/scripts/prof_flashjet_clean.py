"""Clean Nsight Systems driver: _spin clock-ramp and JIT warmup happen BEFORE
cudaProfilerStart, so with `nsys profile --capture-range=cudaProfilerApi` the
report contains only flashjet's real kernels (no volta_sgemm warmup artifact).
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

p4, mask = make_batch(B, N, "cuda"); p4 = p4.float()
_spin(p4.device)  # clock ramp — OUTSIDE the capture range
cluster_batch_triton_large(p4, mask, 0.4, -1.0)               # JIT warmup
flashjet.cluster(p4, mask, R=0.4, algorithm="antikt", backend="triton-large")
torch.cuda.synchronize()

torch.cuda.profiler.start()   # <-- nsys capture begins here
for i in range(ITERS):
    nvtx.range_push(f"kernel-only/iter{i}")
    cluster_batch_triton_large(p4, mask, 0.4, -1.0)
    torch.cuda.synchronize(); nvtx.range_pop()
for i in range(ITERS):
    nvtx.range_push(f"api+decode/iter{i}")
    flashjet.cluster(p4, mask, R=0.4, algorithm="antikt", backend="triton-large")
    torch.cuda.synchronize(); nvtx.range_pop()
torch.cuda.profiler.stop()    # <-- capture ends
print(f"profiled (clean) B={B} N={N} iters={ITERS}")
