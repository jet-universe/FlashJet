# flashjet benchmarking & profiling results

All measured on **Tesla T4** (driver 580, CUDA 13), anti-kt R=0.4, float32 GPU
backends. flashjet code state: branch `audit-remediation`. FastJet baseline:
scikit-hep **fastjet 3.5.1.3** (reference validated against this exact build:
`test_reference_vs_fastjet.py` + `test_kinematics_vs_fastjet.py` → 13 passed).

> Note: the T4 is a modest GPU; an A100/H100 widens every GPU margin below.

## 1. GPU backend sweep (synthetic events, B=1024)

| N | torch | fused triton | triton-large | triton-large via api |
|---|---|---|---|---|
| 32 | 58.1 ms | 2.98 ms (20×) | 1.53 ms (38×) | 1.76 ms |
| 64 | 183.8 ms | 22.6 ms (8×) | 1.14 ms (162×) | 1.38 ms |
| 128 | 1160.5 ms | 68.1 ms (17×) | 6.61 ms (176×) | 4.89 ms |

Large-N (triton-large only; torch impractical):

| N | B | ms/batch | µs/event |
|---|---|---|---|
| 512 | 512 | 25.6 | 49.9 |
| 2048 | 256 | 86.6 | 338 |
| 6000 | 128 | 418 | 3267 |
| 16384 | 64 | 2011 | 31416 |

On the T4 triton-large beats the fused kernel even at N=32 — the auto-dispatch
crossover sits lower than the A100-tuned 32–64.

## 2. Standalone bench vs FastJet (synthetic, matched sizes)

| N | FastJet CPU µs/evt (classic) | flashjet GPU µs/evt | speedup |
|---|---|---|---|
| 32 | 123 | 1.7 | ~72× |
| 64 | 240 | 1.3 | ~180× |
| 128 | 477 | 4.8 | ~100× |
| 512 | 2011 | 29 | ~70× |
| 2048 | 8530 | 342 | ~25× |
| 6000 | 25620 | 3289 | ~8× |

## 3. Apples-to-apples on REAL events (16384 ttbar jets, identical inputs)

Source: `/eos/user/c/cgupta/training-samples/hlt/out_TT_11.root`, tree
`DeepJetNTupler/DeepJetvars`, branches `jet_pfcand_{pt,eta,phi,mass}`.
193,400 constituents, mean 11.8/jet.

| clusterer | µs/jet | Mpart/s | speedup vs flashjet |
|---|---|---|---|
| **flashjet GPU (T4, auto)** | **0.94** | **12.5** | — |
| FastJet awkward (CPU vectorized) | 10.27 | 1.15 | ~11× |
| FastJet classic (CPU per-event) | 60.25 | 0.20 | ~64× |

**n_jets agreement flashjet vs FastJet: 16384/16384 = 100.00%.**

Fair Python CPU baseline is the awkward (vectorized) interface → **~11×**.
Classic is the validation reference. Production target = C++ FastJet in CMSSW.

## 4. Profiling — nsys (triton-large, B=256, N=2048)

Clean capture (warmup/`_spin` excluded via cudaProfilerApi range):

| kernel | % GPU time | avg/call |
|---|---|---|
| `_cluster_large_kernel` | 99.3% | 78.4 ms |
| `_decode_kernel` | 0.5% | 0.42 ms |
| torch validate/glue | <0.2% | µs |

No host↔device traffic in steady state (10 KB D2H + 5 KB memset / 10 iters).
The single serial kernel dominates; the single-launch decode is effectively free.

## 5. Profiling — ncu (`_cluster_large_kernel`, B=128, N=512)

| metric | value |
|---|---|
| Duration | 11.31 ms |
| DRAM throughput | 0.36% |
| L1/TEX cache throughput | 57.4% |
| Compute (SM) throughput | 39.4% |
| Registers/thread | 168 |
| Theoretical occupancy | 37.5% (register-limited, 3 blocks/SM) |
| Achieved occupancy | 33.2% |
| Waves/SM | 1.07 |

At this size the kernel is **not DRAM-bandwidth bound** (scratch fits in cache);
it is **L1-cache-traffic + occupancy/latency bound**. High register pressure
(168/thread) caps occupancy at 37.5% and only ~1 wave/SM fills the GPU once.
→ optimization levers: reduce register pressure, raise parallelism (larger
batch, large-N tail handoff to the register-resident kernel). Caveat: re-check
at much larger N where scratch spills past cache and DRAM may dominate.
