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

## Baseline runs (before optimization)

2D baseline:

    python scripts/bench_wave.py --dim 2 --sizes 11,21,31,41 --steps 50 --repeats 3 --dt 0.1 | tee bench_2d_before.csv

3D baseline:

    python scripts/bench_wave.py --dim 3 --layers 5 --sizes 7,11,15 --steps 30 --repeats 3 --dt 0.1 | tee bench_3d_before.csv

Notes:
- If stability_adjusted=1, dt was clamped internally. For fair comparisons, use the same effective_dt.

## Results

### 2D (steps=50, repeats=3, dt=0.1)

| size | nodes | edges | steps | repeats | sim_s_median | effective_dt | stability_adjusted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 484 | 5,566 | 50 | 3 | 0.051075 | 0.100000 | 0 |
| 21 | 1,764 | 37,926 | 50 | 3 | 0.278252 | 0.100000 | 0 |
| 31 | 3,844 | 121,086 | 50 | 3 | 0.820960 | 0.100000 | 0 |
| 41 | 6,724 | 279,046 | 50 | 3 | 1.828803 | 0.100000 | 0 |

### 3D (layers=5, steps=30, repeats=3, dt=0.1)

| size | layers | nodes | edges | steps | repeats | sim_s_median | effective_dt | stability_adjusted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 5 | 980 | 8,134 | 30 | 3 | 0.057394 | 0.100000 | 0 |
| 11 | 5 | 2,420 | 29,766 | 30 | 3 | 0.182789 | 0.100000 | 0 |
| 15 | 5 | 4,500 | 73,350 | 30 | 3 | 0.402404 | 0.100000 | 0 |

## Bottleneck hypothesis

run_wave_sim iterates over nodes and sums neighbor values each step, so runtime is expected to be dominated by Python-level loops for larger graphs.

## Optimization plan (next)

- Change:
- Before/after command:
- Before/after tables:
- Correctness checks run (pytest / invariants):
