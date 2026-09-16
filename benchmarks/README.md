# benchmarks/

Standalone benchmarking + profiling of flashjet vs FastJet, run on a Tesla T4.
Headline: on identical real CMS ttbar jets, flashjet is **~11× faster than
vectorized FastJet** (CPU) with **100% n_jets agreement**. See
[results/RESULTS.md](results/RESULTS.md) for all numbers.

## Environments (node-specific; see project memory)

- **GPU backends** run in the `b_hive` micromamba env (torch 2.5.1 + triton 3.1
  + flashjet editable). Do NOT install fastjet there (breaks its awkward pin).
- **FastJet baseline** runs in a separate `fjbench` env
  (`conda-forge python fastjet numpy libstdcxx-ng` + `pip install fastjet`,
  scikit-hep 3.5.1.3). Isolated so it can't disturb b-hive.

Scripts use absolute paths (env python, the real root file, /tmp scratch) as
actually run; adjust for another node.

## Scripts (`scripts/`)

| script | env | what |
|---|---|---|
| `bench_large.py` | b_hive | triton-large throughput at large N |
| `fj_baseline.py` | fjbench | FastJet classic CPU baseline (synthetic) |
| `real_extract_flashjet.py [NJETS]` | b_hive | read real ttbar jets, time flashjet, save events to /tmp |
| `real_fastjet.py` | fjbench | cluster the SAME saved jets with FastJet classic + awkward |
| `real_events.py` | b_hive | single-script real-event run + reference cross-check |
| `prof_flashjet_clean.py` | b_hive | nsys driver, warmup outside cudaProfilerApi capture range |
| `prof_flashjet.py` | b_hive | nsys driver (simple; includes _spin SGEMM artifact) |
| `prof_ncu.py` | b_hive | one `_cluster_large_kernel` launch for Nsight Compute |
| `run_nsys.sh [B N iters]` | — | capture timeline + dump kernel/mem/NVTX tables |

`scripts/bench_gpu.py` / `bench_vs_fastjet.py` referenced in RESULTS are the
repo's existing `../scripts/` benchmarks.

## Profiling commands

```bash
NSYS=/opt/nvidia/nsight-systems/2025.6.3/bin/nsys   # CUDA-bundled stubs are broken
PY=.../envs/b_hive/bin/python                        # point nsys at the env python

# clean timeline (excludes _spin warmup)
$NSYS profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o results/flashjet_clean_B256_N2048 \
  $PY scripts/prof_flashjet_clean.py 256 2048 10

# Nsight Compute SOL/occupancy on one kernel launch
ncu --kernel-name "regex:_cluster_large_kernel" --launch-count 1 --set basic \
  -o results/flashjet_ncu_B128_N512 $PY scripts/prof_ncu.py 128 512
```

## Results (`results/`)

- `RESULTS.md` — consolidated tables + interpretation
- `raw_logs.txt` — raw console outputs
- `nsys_clean_kernel_summary.txt` — kernel table from the nsys capture above
- `ncu_cluster_large_details.txt` — SOL/occupancy from the ncu capture above

The binary captures themselves (`*.nsys-rep`, `*.ncu-rep`) are gitignored, as
are real-event `*.npy` dumps (derived from CMS data). All of it is regenerable
from the commands above on a GPU box; the text summaries extracted from them
are what is worth keeping in the repo.
