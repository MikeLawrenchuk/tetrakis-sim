from __future__ import annotations

import json
from pathlib import Path

from tetrakis_sim.evals.cli import main as eval_main


def test_evals_cli_smoke(tmp_path: Path) -> None:
    out = tmp_path / "dc.jsonl"

    rc = eval_main(
        [
            "generate",
            "--out",
            str(out),
            "--n-per-class",
            "1",
            "--seed",
            "1",
            "--size",
            "7",
            "--dim",
            "2",
            "--steps",
            "10",
            "--dt",
            "0.1",
        ]
    )
    assert rc == 0
    assert out.exists()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # 4 classes * n-per-class=1

    rec0 = json.loads(lines[0])
    assert rec0["schema_version"] == 1
    assert "features" in rec0
    assert "params" in rec0
    assert "dt_requested" in rec0["params"]
    assert "dt_used" in rec0["params"]

    rc2 = eval_main(["baseline", "--data", str(out), "--seed", "1", "--test-frac", "0.5"])
    assert rc2 == 0
