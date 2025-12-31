# Performance

This note records benchmark commands and measured results for build_sheet and run_wave_sim.

## Benchmark script

The benchmark prints CSV (one row per lattice size):

    python scripts/bench_wave.py --help

Canonical usage (write outputs under `benchmarks/`):

    mkdir -p benchmarks
    python scripts/bench_wave.py --dim 2 --sizes 11,21,31,41 --steps 50 --repeats 3 --dt 0.1 | tee benchmarks/bench_2d_before.csv

The output is CSV with one row per lattice size.

## Methodology

- Timings use `time.perf_counter()` inside `scripts/bench_wave.py`.
- For each lattice size: run `repeats=3` and report `sim_s_median` (CSV also includes `sim_s_min` / `sim_s_max`).
- If `stability_adjusted=1`, the simulator clamped `dt` internally. Compare runs at matched `effective_dt`.

## Machine

Fill this in when you record results:

- Git commit: bdd46d111546eda03891eb7d3e83e8fa8c42199a
- numpy: 1.26.4
- networkx: 3.4.2
- Model: MacBook Pro (MacBookPro17,1)
- CPU: Apple M1
- RAM: 8 GB
- OS: macOS 13.5 (22G74)
- Python: 3.11.9
- Install method: venv (.venv), pip install -e ".[dev]"


## Baseline runs (before optimization)

2D baseline:

    python scripts/bench_wave.py --dim 2 --sizes 11,21,31,41 --steps 50 --repeats 3 --dt 0.1 | tee benchmarks/bench_2d_before.csv

3D baseline:

    python scripts/bench_wave.py --dim 3 --layers 5 --sizes 7,11,15 --steps 30 --repeats 3 --dt 0.1 | tee benchmarks/bench_3d_before.csv

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

## Optimization implemented

**Change:** precompute neighbor lists, degrees, and per-node mass/potential once per run; reduce repeated NetworkX lookups in the inner loop.

**Observed speedup (median sim time):** 2D 1.36x–1.51x; 3D 1.50x–1.77x.

### After optimization (2D; steps=50, repeats=3, dt=0.1)

| size | nodes | edges | steps | repeats | sim_s_median | speedup_vs_baseline | effective_dt | stability_adjusted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 484 | 5,566 | 50 | 3 | 0.033724 | 1.51x | 0.100000 | 0 |
| 21 | 1,764 | 37,926 | 50 | 3 | 0.191302 | 1.45x | 0.100000 | 0 |
| 31 | 3,844 | 121,086 | 50 | 3 | 0.583171 | 1.41x | 0.100000 | 0 |
| 41 | 6,724 | 279,046 | 50 | 3 | 1.346898 | 1.36x | 0.100000 | 0 |

### After optimization (3D; layers=5, steps=30, repeats=3, dt=0.1)

| size | layers | nodes | edges | steps | repeats | sim_s_median | speedup_vs_baseline | effective_dt | stability_adjusted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 5 | 980 | 8,134 | 30 | 3 | 0.032374 | 1.77x | 0.100000 | 0 |
| 11 | 5 | 2,420 | 29,766 | 30 | 3 | 0.113395 | 1.61x | 0.100000 | 0 |
| 15 | 5 | 4,500 | 73,350 | 30 | 3 | 0.268107 | 1.50x | 0.100000 | 0 |
