# Deep Dive Packet — tetrakis-sim

This document is a reviewer/interviewer-friendly entry point to the repository: what it does, how it is structured, how it is validated, how it is evaluated, and how to reproduce key results.

## What this project does

`tetrakis-sim` is a research-oriented simulator for wave propagation on a discrete medium represented as a graph (a “tetrakis” lattice). It supports defect operators that modify graph topology and/or node attributes, and it provides:

- Lattice generation (2D/3D)
- Defects (geometry/topology + node attributes)
- Explicit wave simulation with a CFL-like stability clamp
- FFT-based spectral analysis and feature extraction
- A reproducible evaluation harness (dataset generation + baseline scoring)
- Performance benchmarking + a measured optimization

## Quickstart (developer)

From the repository root:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit run --all-files
python -m pytest
```

## Reproduce the evaluation harness

### 1) Generate a labeled dataset (JSONL)

```bash
tetrakis-eval generate \
  --out /tmp/defect_classification.jsonl \
  --n-per-class 20 \
  --seed 0 \
  --size 11 \
  --dim 2 \
  --layers 1 \
  --steps 40 \
  --dt 0.2 \
  --damping 0.0
```

Each JSONL record contains:

* `schema_version`, `id`, `task`, `label`
* `params` (including `dt_requested` and `dt_used`)
* `features` (graph + spectral + time-series features)

### 2) Run the baseline classifier

```bash
tetrakis-eval baseline \
  --data /tmp/defect_classification.jsonl \
  --seed 0 \
  --test-frac 0.3 \
  --split stratified \
  --out-metrics /tmp/baseline_metrics.json
```

The baseline prints metrics to stdout and optionally writes JSON metrics to `--out-metrics`.

### 3) Export an LLM-eval JSONL (no API calls)

`llm-export` converts the dataset into `{prompt, expected}` lines suitable for downstream LLM evaluation runners.

```bash
tetrakis-eval llm-export \
  --data /tmp/defect_classification.jsonl \
  --out /tmp/llm_eval.jsonl \
  --format kv \
  --max-features 0
```

This step does not call any model and does not require any API key.

## Reproduce benchmarks (performance/scaling)

Benchmark lattice construction + wave simulation scaling:

```bash
mkdir -p benchmarks
python scripts/bench_wave.py --dim 2 --sizes 11,21,31,41 --steps 50 --repeats 3 --dt 0.1 | tee benchmarks/bench_2d.csv
python scripts/bench_wave.py --dim 3 --layers 5 --sizes 7,11,15 --steps 30 --repeats 3 --dt 0.1 | tee benchmarks/bench_3d.csv
```

For methodology and recorded results, see:

* `docs/performance.md`

## Architecture

### Pipeline view

```text
build_sheet(...)                # lattice generation
  -> apply_defect(...)          # topology/attribute changes
    -> run_wave_sim(...)        # explicit time stepping with stability clamp
      -> run_fft(...)           # spectrum from node time series
        -> feature extraction   # eval features from spectrum/time-series/graph stats
          -> baseline + metrics # nearest centroid scoring (and llm-export)
```

### Key modules

* `tetrakis_sim/lattice.py`

  * `build_sheet(...)` builds a 2D or 3D lattice graph.
* `tetrakis_sim/defects.py`

  * `apply_defect(...)` applies defect operators (blackhole/wedge/singularity).
* `tetrakis_sim/physics.py`

  * `run_wave_sim(...)` runs the explicit wave update and clamps `dt` if unstable.
  * `run_fft(...)` computes FFT of a node’s time series.
* `tetrakis_sim/evals/`

  * `dataset.py`: generates labeled JSONL datasets
  * `features.py`: extracts spectral/time-series features
  * `baseline.py`: nearest-centroid baseline + metrics
  * `llm_export.py`: exports prompt/expected JSONL for LLM evals
  * `cli.py`: `tetrakis-eval` CLI entrypoint
* `scripts/bench_wave.py`

  * scaling benchmark harness

## Model and validation

For the conceptual model summary and validation checklist, see:

* `docs/model_and_validation.md`

### What is validated in tests

The test suite includes “golden” invariants that assert:

* defect structural semantics (node/edge effects + attribute tagging)
* stability clamp behavior (`stability_adjusted`, `effective_dt`, and warning capture)
* eval dataset schema/feature contracts

These tests are designed to fail fast if core semantics change.

## Notable engineering decisions

* **Stability guardrail:** `dt` is clamped using a max-degree CFL-like criterion, and the effective `dt` is recorded in history metadata for reproducibility.
* **Measured optimization:** the wave inner loop reduces NetworkX overhead by precomputing neighbor lists, degrees, and static node attributes.
* **Reproducible eval artifacts:** dataset JSONL + optional metrics JSON are stable, versionable artifacts.

## Next steps

High-impact extensions (in rough priority order):

1. Add one more eval task (e.g., severity/radius regression or anomaly detection)
2. Add a fast array backend for `run_wave_sim` (node→index mapping) and benchmark it
3. Docs automation (Sphinx/MkDocs) + CI docs build
4. Registry/extensibility scaffold for adding new lattice/defect types without editing core code
