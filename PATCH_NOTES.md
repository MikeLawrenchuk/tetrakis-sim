# Tetrakis-Sim Roadmap Add-ons (Singularity + Curvature)

This bundle contains **new files** you can drop into your repo:

- `tetrakis_sim/curvature.py` : minimal discrete-curvature metrics (angle deficit)
- `examples/naked_singularity_sweep.py` : canonical sweep runner comparing defect regimes
- `notebooks/naked_singularity_demo.ipynb` : notebook wrapper around the sweep

## 1) README edits (recommended)

In `README.md`, update the CLI defect list to include singularity.

Current section (around the CLI arguments list) shows:

    --defect_type: blackhole, wedge, none

Change to:

    --defect_type: blackhole, wedge, singularity, none

Then add a short "Singularity / naked singularity" section (example text below):

### Singularity / Naked Singularity (Analogue)

`--defect_type singularity` models a *naked singularity analogue* by keeping the center node(s)
present (no full event horizon) while increasing local curvature and/or stiffening local dynamics.

Example:

    python scripts/run_batch.py --size 11 --layers 5 --steps 60 \
      --defect_type singularity --radius 0.5 \
      --sing_mass 200 --sing_potential 25

See:
- `examples/naked_singularity_sweep.py`
- `notebooks/naked_singularity_demo.ipynb`

If your CLI uses different flag names (e.g. `--singularity_mass`), mirror those.

## 2) Integrate curvature metrics into metadata (optional but very aligned with the roadmap)

In your batch runner (likely `scripts/run_batch.py`), right after:
- building the lattice graph, and
- applying the defect,

compute curvature summaries and inject them into your metadata JSON.

Example snippet:

    from tetrakis_sim.curvature import curvature_report

    # after defect applied:
    curv = curvature_report(G, hops=2)
    metadata.update(curv)  # or metadata["curvature"]=curv["curvature"]

This produces JSON fields like:
    metadata["curvature"]["global"]["mean"], ...

Note: this expects node positions to exist on your lattice nodes (e.g. `pos=(x,y)`).

## 3) Run the canonical sweep

From the repo root:

    python examples/naked_singularity_sweep.py --help
    python examples/naked_singularity_sweep.py --size 11 --layers 3 --steps 40

Outputs go into `batch_cli_output_singularity/` by default.

If your CLI does not support `--sing_mass` / `--sing_potential`, the sweep script will detect that
via `--help` and simply skip those options.
