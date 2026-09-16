import sys, time, torch, flashjet
import sys; sys.path.insert(0,"scripts"); from bench_gpu import make_batch
from flashjet.triton_large import cluster_batch_triton_large
from flashjet.tune import _spin

def bench(fn, iters=10):
    fn(); torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters

torch.cuda.init(); 
for N,B in [(512,512),(2048,256),(6000,128),(16384,64)]:
    p4,mask=make_batch(B,N,"cuda"); p4=p4.float(); _spin(p4.device)
    t=bench(lambda: cluster_batch_triton_large(p4,mask,0.4,-1.0))
    ta=bench(lambda: flashjet.cluster(p4,mask,R=0.4,algorithm="antikt",backend="triton-large"))
    print(f"N={N:>6} B={B:>4}: triton-large {t*1e3:8.2f} ms/batch  {t/B*1e6:8.1f} us/event | via api {ta*1e3:8.2f} ms", flush=True)
