# A100 profiling — event-based vs jet-based clustering

GPU: **NVIDIA A100-PCIE-40GB**, triton 3.6, CUDA 13. anti-kt R=0.4, via
`flashjet.cluster` (post dispatch-crossover fix). DVFS-spun before timing.

- **Event-based** = a full event, large N (one Triton program / event): **B=256, N=2048**.
- **Jet-based** = recluster a jet's constituents, small N / large B: **B=16384, N=64**.

Artifacts in this dir: `timing_decomposition.txt`. The nsys captures these
numbers came from (`event_B256_N2048.nsys-rep`, `jet_B16384_N64.nsys-rep`) are
gitignored — regenerate with `benchmarks/scripts/run_nsys.sh` on an A100.

## 1. Timing decomposition (wall, median)

| stage | EVENT (B=256,N=2048) | JET (B=16384,N=64) |
|---|---|---|
| **full `cluster()`** | **23.2 ms** (90.6 µs/event) | **5.60 ms** (0.34 µs/jet) |
| validate sync | 0.15 ms (0.6%) | 0.12 ms (2.2%) |
| kernel + host + launch | 23.0 ms (99.1%) | 5.41 ms (96.5%) |
| decode (jet_idx) | 0.15 ms (0.7%) | 0.16 ms (2.9%) |

Fixed costs (validate, decode) are a bigger *fraction* in the jet regime only because
the kernel itself is ~4× faster there — in absolute terms they are ~constant (~0.15 ms).

## 2. nsys — GPU kernel summary (10 clean iters, capture-range)

**EVENT (B=256, N=2048):**

| kernel | % GPU | avg/call |
|---|---|---|
| `_cluster_large_kernel` | **99.6%** | 22.3 ms |
| `_decode_kernel` | 0.3% | 69.5 µs |
| torch glue (validate etc.) | <0.1% | µs |

**JET (B=16384, N=64):**

| kernel | % GPU | avg/call |
|---|---|---|
| `_cluster_large_kernel` | **97.5%** | 5.26 ms |
| `_decode_kernel` | 1.5% | 78.7 µs |
| torch glue (validate etc.) | ~1.0% | µs |

A100 `_cluster_large_kernel` at B=256/N=2048 is **22.3 ms** vs the T4's 78.4 ms
(`../nsys_clean_kernel_summary.txt`) — ~3.5× faster, same 99%+ dominance.

## 3. ncu — kernel deep-dive  ⚠️ BLOCKED

ncu needs the GPU hardware performance counters, which are held by this node's DCGM
monitoring (`nv-hostengine` + `dcgm-exporter` running). `RmProfilingAdminOnly: 0`, so
it is **not** a permissions issue — purely a concurrent-collector conflict. ncu errors:
"a driver resource was unavailable … no other tool (like DCGM)".

To collect: pause DCGM, run ncu, resume —
```
dcgmi profile --pause
ncu --set basic -k regex:cluster_large -c 1 python benchmarks/scripts/prof_ncu.py 256 2048
ncu --set basic -k regex:cluster_large -c 1 python benchmarks/scripts/prof_ncu.py 16384 64
dcgmi profile --resume
```
(The T4 ncu of the same kernel — 168 reg/thread, 37.5% occupancy, register-limited —
is in `../ncu_cluster_large_details.txt`, but that does not transfer to the A100.)
