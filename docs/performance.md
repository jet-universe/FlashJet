# Performance and benchmarks

Measure the whole workload you intend to run. Kernel time alone omits padding,
host-to-device copies, decoding, and jet recovery. Small batches may leave most
of the GPU idle, while excessive padding increases work and memory use.

The repository preserves the original benchmark records under
[benchmarks](https://github.com/jet-universe/FlashJet/tree/main/benchmarks).
They describe particular machines and software versions. They are not release
performance guarantees, and were not rerun during this publication snapshot.

| Script | Measures |
|---|---|
| `scripts/bench_gpu.py` | GPU backends |
| `scripts/bench_vs_fastjet.py` | FlashJet versus FastJet Python interfaces |
| `scripts/bench_pipeline.py` | Data loading and GPU pipeline |
| `scripts/bench_cpp.sh` | Both CPU implementations in C++ |
| `scripts/tune_large.py` | Large-kernel launch settings |

Run scripts from the repository root. The C++ comparison fetches/builds FastJet
and needs a compiler and network access. Check each script's arguments before
starting a long run. Warm up the GPU and separate compilation from steady-state
timing; clocks and one-time allocations can distort short measurements.

## Optional tuning

`FLASHJET_TUNE=1` benchmarks candidate large-kernel configurations on the first
call for a GPU model and size band. Results are saved in
`~/.cache/flashjet/tune.json`; `FLASHJET_TUNE_CACHE` overrides the file path.
Retune after a driver or Triton upgrade by moving the old cache aside. Different
configurations can change decisions near ties. Tuning is off by default.

`FLASHJET_COMPILE_DECODE=1` enables `torch.compile` for history decoding. Its
first call compiles for the tensor shape and can be slow. Measure whether it
helps your full pipeline before enabling it.
