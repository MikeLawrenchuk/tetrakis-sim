# Extensibility Architecture (Registry Scaffold)

## Goal

Allow new lattices, defects, and evaluation tasks to be added without editing multiple CLI modules or large `if/elif` blocks.

## Registry concept

`tetrakis_sim/registry.py` provides three registries:

- `lattice`: graph builders (e.g., tetrakis 2D/3D sheet builders)
- `defect`: defect operators (e.g., wedge, blackhole, singularity)
- `eval_task`: dataset generation tasks (e.g., defect classification, radius regression)

Each registry maps a stable string key (name) to a callable (`fn`) plus optional metadata (`description`).

## Minimal API

- `register(kind, name, fn, description=..., overwrite=False)`
- `get(kind, name) -> fn`
- `list_names(kind) -> [name, ...]`
- Convenience wrappers:
  - `register_lattice(name, fn, ...)`
  - `register_defect(name, fn, ...)`
  - `register_eval_task(name, fn, ...)`

## Intended integration points

In future refactors:

- CLIs should map command-line choices to registry keys.
- Example:
  - `--defect wedge` selects `registry.get("defect", "wedge")`
  - `--task defect_classification` selects `registry.get("eval_task", "defect_classification")`

This keeps the CLI stable while allowing the implementation set to grow.

## Next refactor step (optional)

- Register the existing built-in implementations at import time:
  - lattices: `build_sheet`
  - defects: `apply_defect`
  - eval tasks: `generate_defect_classification_jsonl`
- Update `tetrakis-batch` and `tetrakis-eval` to use `list_names(...)` for argparse `choices=...`.
