# Project Review

## Snapshot
- **Domain fit:** Discrete lattice generation (`tetrakis_sim.lattice.build_sheet`) and defect modeling (`tetrakis_sim.defects.apply_wedge_defect`/`apply_blackhole_defect`) are implemented as small, composable helpers, which keeps physics/CLI code straightforward to follow.
- **User entry points:** The CLI (`tetrakis_sim.cli.main`) builds lattices, applies defects, optionally runs wave simulations/FFTs, and can render plots in one go; the library API re-exports the most approachable helpers via `tetrakis_sim.__init__`.
- **Tests:** `pytest` passes end-to-end and exercises the lattice, defect, physics, and plotting stubs (warnings appear because optional plotting backends are intentionally missing in the environment).

## Strengths
- **Clear lattice primitives:** The lattice builder is concise and symmetric between 2-D and 3-D grids, with a small `clique` helper that makes the construction logic easy to reason about.
- **Defect ergonomics:** Defect helpers normalize outputs through `DefectResult`, so callers get consistent metadata even when only the graph mutation is needed.
- **Optional plotting stubs:** Plotting utilities degrade gracefully when matplotlib/plotly are absent, allowing tests and headless environments to import the package without hard dependencies.

## Opportunities
- **Optional dependencies:** The import-time warnings from the plotting stubs are noisy during tests; consider gating them behind logging or a dedicated `extra_requires` entry so users can install visualization extras and avoid the warnings when desired.
- **CLI defaults:** The CLI currently builds a 2-D lattice and always runs a wave simulation when `--physics` is `wave` or `fft`; offering presets (e.g., lighter grids for demos) and progress output would make batch runs friendlier.
- **Documentation depth:** The README walks through usage, but a brief architecture section describing the lattice/defect/physics modules and their expected node tuple shapes would help new contributors navigate the codebase more quickly.

## Quick Wins
- Add a `plot` extra to `setup.py`/`requirements.txt` that pulls in matplotlib+plotly, and demote the warnings to `logging.warning` so default imports stay quiet.
- Expand `scripts/run_batch.py --help` with a couple of concrete invocations for common parameter sweeps (e.g., 2-D vs 3-D, wedge vs black hole) to mirror the README examples.
- Create a short “module tour” in `docs/` that explains how lattices, defects, and physics components interact; this would give readers a low-friction starting point before diving into notebooks.
