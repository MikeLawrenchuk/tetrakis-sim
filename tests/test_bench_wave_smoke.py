import os
import subprocess
import sys
from pathlib import Path


def test_bench_wave_script_runs_and_emits_csv():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "bench_wave.py"

    cmd = [
        sys.executable,
        str(script),
        "--dim",
        "2",
        "--sizes",
        "5",
        "--steps",
        "2",
        "--repeats",
        "1",
        "--dt",
        "0.05",
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

    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")

    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    assert lines, "Expected CSV output."
    assert lines[0].startswith("dim,layers,size,"), f"Unexpected header: {lines[0]}"
    assert len(lines) >= 2, "Expected at least one data row."
