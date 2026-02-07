from __future__ import annotations

import json
from pathlib import Path

from tetrakis_sim.evals.cli import main as eval_main


def test_llm_export_smoke(tmp_path: Path) -> None:
    data = tmp_path / "dc.jsonl"
    out = tmp_path / "llm.jsonl"

    rc = eval_main(
        [
            "generate",
            "--out",
            str(data),
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
    assert data.exists()

    rc2 = eval_main(
        [
            "llm-export",
            "--data",
            str(data),
            "--out",
            str(out),
            "--format",
            "kv",
        ]
    )
    assert rc2 == 0
    assert out.exists()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    rec0 = json.loads(lines[0])
    assert "prompt" in rec0
    assert "expected" in rec0
    assert "Possible labels:" in rec0["prompt"]
    assert rec0["expected"] in {"none", "wedge", "blackhole", "singularity"}
