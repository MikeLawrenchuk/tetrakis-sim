# Model and Validation (Tetrakis-Sim)

This note documents (1) what the simulator is computing and (2) how to validate it in a repeatable way.

## Model summary

### State space

Tetrakis-Sim represents a discrete medium as a graph **G = (V, E)**:

- **Nodes** encode lattice locations (2D or 3D) plus an internal quadrant label (e.g., `A/B/C/D`).
- **Edges** encode couplings/constraints between nodes.

### Defects (geometry changes)

Defects modify the graph (topology and/or node attributes):

- **none**: no change
- **wedge**: removes a specific local edge at the center (a deficit)
- **blackhole**: removes all nodes within a radius of the center
- **singularity**: does not remove nodes; tags nodes near the center with `mass`, `potential`, optional local edge pruning

### Dynamics (discrete wave update)

Wave propagation is simulated as an explicit time-stepping update over the graph (Laplacian-driven coupling).
Conceptually:

- `Δu(n) = Σ_{nb∈N(n)} u(nb) − deg(n)·u(n)`
- the solver updates node values over time using `c` and `dt`, with optional damping
- `mass` / `potential` node attributes (used by the singularity defect) modulate the local response

The implementation may clamp `dt` to avoid instability on dense graphs.

## Observable outputs

### Time series

`run_wave_sim(...)` produces a time series of node values.

### Spectrum

`run_fft(history, node=...)` returns a frequency axis and amplitude spectrum (plus the node’s time series).
Features derived from FFT are used for evaluation.

## Validation checklist

### 1) Structural validation (graph-level)

For a fixed `size/dim/layers/defect params`:

- **blackhole**: removed node count should be consistent for the same radius/center
- **wedge**: exactly one local edge removal at the intended location (2D) or intended layer (3D)
- **singularity**: node attribute tagging matches radius; optional pruning removes intended local edges

### 2) Numerical sanity checks (simulation-level)

- Run with a fixed kick node and confirm the simulation remains bounded for default settings.
- Confirm `dt` clamping (if triggered) produces a non-blowing-up run.

### 3) End-to-end reproducibility (evaluation harness)

The eval harness provides a repeatable “generate → score” loop.

Use the sanity check below (`n-per-class >= 10`) to validate the full “generate → score” loop.

#### Evaluation harness sanity check

Generate a dataset (use enough samples so the train/test split includes all classes):

```bash
tetrakis-eval generate --out /tmp/defect_classification.jsonl --n-per-class 20 --size 11 --layers 1 --steps 40 --dim 2 --seed 0
```

Score it with the baseline:

```bash
tetrakis-eval baseline --data /tmp/defect_classification.jsonl --test-frac 0.3 --seed 0
```

This validates the full pipeline:
lattice → defect → wave → FFT → features → baseline scoring.

Notes:

- Very small datasets can produce misleading accuracy (including 0.0) if the train split does not contain every class.
- Use `n-per-class >= 10` for sanity checks.

What to improve next:

- Add one “golden” configuration in tests (fixed seed) whose key metrics are asserted (counts + a small set of features).
- Expand the eval task set beyond defect classification (e.g., radius regression; layer classification for 3D).
