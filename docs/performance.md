# Performance

This note records benchmark commands and measured results for build_sheet and run_wave_sim.

## Benchmark script

Command:

    python scripts/bench_wave.py | tee bench.csv

The output is CSV with one row per lattice size.

## Machine

Fill this in when you record results:

- Model:
- CPU:
- RAM:
- OS:
- Python:
- Install method: (e.g., pip install -e ".[dev]")

## Recommended baseline runs

2D baseline:

    python scripts/bench_wave.py --dim 2 --sizes 11,21,31,41 --steps 50 --repeats 3 --dt 0.1 | tee bench_2d_before.csv

3D baseline:

    python scripts/bench_wave.py --dim 3 --layers 5 --sizes 7,11,15 --steps 30 --repeats 3 --dt 0.1 | tee bench_3d_before.csv

Notes:
- If stability_adjusted=1, dt was clamped internally. For fair comparisons, use the same effective_dt.

## Results

### 2D

| size | nodes | edges | steps | repeats | sim_s_median | effective_dt | stability_adjusted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | |

### 3D

| size | layers | nodes | edges | steps | repeats | sim_s_median | effective_dt | stability_adjusted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | | |

## Bottleneck hypothesis

run_wave_sim iterates over nodes and sums neighbor values each step, so runtime is expected to be dominated by Python-level loops for larger graphs.

## Optimization plan (to be filled)

- Change:
- Before/after command:
- Before/after tables:
- Correctness checks run (pytest / invariants):
