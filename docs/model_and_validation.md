# Model and Validation (Tetrakis-Sim)

This note documents (1) what the simulator is computing and (2) how to validate it in a repeatable way.

## Model summary

### Parameters

Common simulation parameters (stored in eval dataset records under `params`):

- `size` (int): lattice width/height in base cells
- `dim` (2|3): spatial dimension
- `layers` (int): number of z-layers when `dim=3` (ignored when `dim=2`)
- `steps` (int): number of update steps
- `c` (float): wave speed parameter
- `dt_requested` (float): timestep requested by the caller
- `dt_used` (float): effective timestep used by the solver after stability clamping
- `damping` (float): discrete damping factor

Defect parameters (task-dependent):

- `blackhole`: `center`, `radius`
- `wedge`: `center` (and `layer` in 3D)
- `singularity`: `center`, `radius`, `mass`, `potential`, `prune_edges`


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

Wave propagation is simulated as an explicit second-order time-stepping update over the graph.

#### Graph coupling term

For a node `n` with neighbor set `N(n)` and degree `deg(n)`, define:

- `lap(n) = Σ_{nb∈N(n)} u_t(nb) − deg(n)·u_t(n)`

(This is `A u − D u`, i.e. the negative of the unnormalized graph Laplacian `L = D − A`.)

#### Update rule (as implemented)

Let `u_t(n)` be the current state and `u_{t-1}(n)` the previous state. The solver uses an effective timestep `dt_eff` (possibly clamped) and:

- `coeff = (c * dt_eff) ** 2`

The update performed at each node is:

- `u_{t+1}(n) = 2*u_t(n) - u_{t-1}(n) + (coeff * inv_mass(n))*lap(n) - (coeff * potential(n))*u_t(n) - damping*(u_t(n) - u_{t-1}(n))`

Where:
- `inv_mass(n) = 1/mass(n)` with any `mass <= 0` treated as `1.0`
- `potential(n)` defaults to `0.0`

These attributes are primarily used by the `singularity` defect.

#### Stability heuristic (CFL-like dt clamp)

The solver computes:

- `max_degree = max_n deg(n)`
- `stability_limit = 1 / sqrt(max_degree)` (only when `max_degree > 0` and `c > 0`)

If `c * dt_requested > stability_limit`, it clamps:

- `dt_eff = stability_limit / c`

A `RuntimeWarning` is emitted when clamping occurs.

It records:
- `history.dt == dt_eff`
- `history.metadata["stability_adjusted"]` (bool)
- `history.metadata["stability_limit"]`
- `history.metadata["effective_dt"]`

The eval dataset generator stores both:
- `params["dt_requested"]` (requested)
- `params["dt_used"]` (effective)

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
tetrakis-eval baseline --data /tmp/defect_classification.jsonl --test-frac 0.3 --seed 0 --split stratified
```

This validates the full pipeline:
lattice → defect → wave → FFT → features → baseline scoring.

Notes:

- Very small datasets can produce misleading accuracy (including 0.0) if the train split does not contain every class.
- Use `n-per-class >= 10` for sanity checks.

What to improve next:

- Add more “golden” configurations and assert a small, stable subset of numeric features (in addition to structural counts).
- Expand the eval task set beyond defect classification (e.g., radius regression; layer classification for 3D).
