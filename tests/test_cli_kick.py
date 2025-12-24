import os
import subprocess
import sys
from pathlib import Path


def test_run_batch_rejects_kick_node_not_in_graph(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    outdir = tmp_path / "out"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_batch.py"),
        "--size", "7",
        "--dim", "2",
        "--layers", "1",
        "--steps", "1",
        "--defect_type", "singularity",
        "--sing_radius", "0.0",
        "--outdir", str(outdir),
        "--kick", '(999,999,"A")',
    ]

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0

    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "--kick node" in combined
    assert "is not in the graph after defect application" in combined
