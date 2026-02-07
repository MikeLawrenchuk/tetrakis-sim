from __future__ import annotations

from pathlib import Path

from tetrakis_sim.evals.baseline import run_nearest_centroid_baseline
from tetrakis_sim.evals.dataset import generate_defect_classification_jsonl


def test_evals_smoke(tmp_path: Path) -> None:
    out = tmp_path / "defect_classification.jsonl"
    generate_defect_classification_jsonl(
        out,
        n_per_class=1,
        seed=0,
        size=7,
        dim=2,
        layers=1,
        steps=5,
        c=1.0,
        dt=0.2,
        damping=0.0,
    )
    metrics = run_nearest_centroid_baseline(out, seed=0, test_frac=0.5, split="stratified")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
