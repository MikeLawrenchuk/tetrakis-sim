#!/usr/bin/env python3
"""examples/naked_singularity_sweep.py

Canonical naked-singularity sweep runner for Tetrakis-Sim.

This script is designed to be "repo-local":
- run it from the repo root (same level as scripts/run_batch.py)
- it calls the existing batch CLI to generate outputs
- it then builds a summary CSV + a simple plot

Because CLI options evolve, the script tries to be resilient:
- it detects which optional flags exist by reading `run_batch.py --help`
- it only passes flags that the CLI actually supports

You should treat this as an experiment harness, not as core library code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunSpec:
    defect_type: str
    radius: float
    sing_mass: Optional[float] = None
    sing_potential: Optional[float] = None


def _run_help(py: str, run_batch: Path) -> str:
    try:
        out = subprocess.check_output([py, str(run_batch), "--help"], text=True, stderr=subprocess.STDOUT)
        return out
    except Exception:
        # If help fails, we still attempt to run with minimal flags.
        return ""


def _supported_flags(help_text: str) -> set:
    # Extract tokens that look like "--flag".
    return set(re.findall(r"--[a-zA-Z0-9_\-]+", help_text))


def _first_supported(flags: Sequence[str], supported: set) -> Optional[str]:
    for f in flags:
        if f in supported:
            return f
    return None


def _dominant_from_spectrum(csv_path: Path) -> Optional[float]:
    try:
        df = pd.read_csv(csv_path)
        # Heuristic column matching
        freq_col = None
        amp_col = None
        for c in df.columns:
            cl = c.strip().lower()
            if "freq" in cl:
                freq_col = c
            if "amp" in cl or "spectrum" in cl:
                amp_col = c
        if freq_col is None:
            freq_col = df.columns[0]
        if amp_col is None:
            amp_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        idx = int(np.argmax(df[amp_col].to_numpy()))
        return float(df[freq_col].iloc[idx])
    except Exception:
        return None


def _dominant_from_metadata(meta_path: Path) -> Optional[float]:
    try:
        meta = json.loads(meta_path.read_text())
        # common key variants
        candidates = [
            "dominant_freq",
            "dominant_frequency",
            "dominantFrequency",
            "peak_freq",
            "peak_frequency",
        ]
        for k in candidates:
            if k in meta:
                return float(meta[k])
        # fall back: scan keys
        for k, v in meta.items():
            kl = str(k).lower()
            if "dominant" in kl and "freq" in kl:
                return float(v)
        return None
    except Exception:
        return None


def run_one(
    py: str,
    run_batch: Path,
    *,
    spec: RunSpec,
    size: int,
    layers: int,
    steps: int,
    outdir: Path,
    c: float,
    dt: float,
    damping: float,
    extra: Sequence[str],
    supported: set,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        py,
        str(run_batch),
        "--size",
        str(size),
        "--layers",
        str(layers),
        "--steps",
        str(steps),
        "--c",
        str(c),
        "--dt",
        str(dt),
        "--damping",
        str(damping),
        "--defect_type",
        spec.defect_type,
        "--radius",
        str(spec.radius),
        "--outdir",
        str(outdir),
    ]

    # Optional singularity flags (try common variants)
    if spec.sing_mass is not None:
        mass_flag = _first_supported(["--sing_mass", "--singularity_mass", "--mass"], supported)
        if mass_flag:
            cmd += [mass_flag, str(spec.sing_mass)]

    if spec.sing_potential is not None:
        pot_flag = _first_supported(["--sing_potential", "--singularity_potential", "--potential"], supported)
        if pot_flag:
            cmd += [pot_flag, str(spec.sing_potential)]

    # Prefix
    prefix = f"{spec.defect_type}_r{spec.radius:g}"
    if spec.sing_mass is not None:
        prefix += f"_M{spec.sing_mass:g}"
    if spec.sing_potential is not None:
        prefix += f"_V{spec.sing_potential:g}"
    cmd += ["--prefix", prefix]

    # Extra user-provided passthrough args
    cmd += list(extra)

    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Locate outputs
    meta_files = sorted(outdir.glob(f"{prefix}*metadata*.json")) + sorted(outdir.glob(f"*{prefix}*metadata*.json"))
    spectrum_files = sorted(outdir.glob(f"{prefix}*spectrum*.csv")) + sorted(outdir.glob(f"*{prefix}*spectrum*.csv"))

    meta_path = meta_files[0] if meta_files else None
    spec_path = spectrum_files[0] if spectrum_files else None

    dom = None
    if meta_path is not None:
        dom = _dominant_from_metadata(meta_path)
    if dom is None and spec_path is not None:
        dom = _dominant_from_spectrum(spec_path)

    return {
        "defect_type": spec.defect_type,
        "radius": float(spec.radius),
        "sing_mass": None if spec.sing_mass is None else float(spec.sing_mass),
        "sing_potential": None if spec.sing_potential is None else float(spec.sing_potential),
        "prefix": prefix,
        "metadata": str(meta_path) if meta_path else None,
        "spectrum": str(spec_path) if spec_path else None,
        "dominant_freq": dom,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run a canonical sweep comparing blackhole/wedge/singularity.")
    p.add_argument("--run_batch", default="scripts/run_batch.py", help="Path to batch runner.")
    p.add_argument("--outdir", default="batch_cli_output_singularity", help="Where to place outputs.")
    p.add_argument("--size", type=int, default=11)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.15)
    p.add_argument("--damping", type=float, default=0.02)

    p.add_argument("--blackhole_radii", default="2.5", help="Comma-separated radii.")
    p.add_argument("--sing_radii", default="0.5,1.0", help="Comma-separated radii (influence or tiny horizon).")

    p.add_argument("--sing_masses", default="50,200", help="Comma-separated masses for singularity (if supported).")

    p.add_argument("--sing_potentials", default="0,25", help="Comma-separated potentials (if supported).")

    p.add_argument(
        "--include",
        default="blackhole,wedge,singularity",
        help="Comma-separated defect types to run.",
    )

    p.add_argument(
        "--extra",
        default="",
        help="Extra args passed through to run_batch (quoted string). Example: '--node 0 --seed 123'.",
    )

    args = p.parse_args(argv)

    run_batch = Path(args.run_batch)
    outdir = Path(args.outdir)

    help_text = _run_help(sys.executable, run_batch)
    supported = _supported_flags(help_text)
    if not supported:
        print("[warn] Could not parse --help; will run with baseline args only.")

    def parse_floats(s: str) -> List[float]:
        return [float(x) for x in s.split(",") if x.strip()]

    blackhole_radii = parse_floats(args.blackhole_radii)
    sing_radii = parse_floats(args.sing_radii)
    sing_masses = parse_floats(args.sing_masses)
    sing_pots = parse_floats(args.sing_potentials)

    include = [x.strip() for x in args.include.split(",") if x.strip()]

    extra = args.extra.strip().split() if args.extra.strip() else []

    runs: List[RunSpec] = []

    if "blackhole" in include:
        for r in blackhole_radii:
            runs.append(RunSpec(defect_type="blackhole", radius=r))

    if "wedge" in include:
        # Wedge defect may ignore radius; we still provide a value for consistency.
        for r in [0.0]:
            runs.append(RunSpec(defect_type="wedge", radius=r))

    if "singularity" in include:
        for r in sing_radii:
            for M in sing_masses:
                for V in sing_pots:
                    runs.append(RunSpec(defect_type="singularity", radius=r, sing_mass=M, sing_potential=V))

    results: List[Dict[str, object]] = []
    for spec in runs:
        rec = run_one(
            sys.executable,
            run_batch,
            spec=spec,
            size=args.size,
            layers=args.layers,
            steps=args.steps,
            outdir=outdir,
            c=args.c,
            dt=args.dt,
            damping=args.damping,
            extra=extra,
            supported=supported,
        )
        results.append(rec)

    df = pd.DataFrame(results)
    summary_csv = outdir / "singularity_sweep_summary.csv"
    df.to_csv(summary_csv, index=False)
    print("\nWrote", summary_csv)

    # Simple plot: dominant frequency vs. mass, faceted by potential, for singularity only
    try:
        df2 = df[df["defect_type"] == "singularity"].dropna(subset=["dominant_freq"])
        if not df2.empty:
            fig = plt.figure()
            for V, sub in df2.groupby("sing_potential"):
                sub = sub.sort_values("sing_mass")
                plt.plot(sub["sing_mass"], sub["dominant_freq"], marker="o", label=f"V={V}")
            plt.xlabel("sing_mass")
            plt.ylabel("dominant_freq")
            plt.title("Singularity sweep: dominant frequency vs mass")
            plt.legend()
            plot_path = outdir / "singularity_sweep_dominant_freq.png"
            plt.savefig(plot_path, dpi=160, bbox_inches="tight")
            print("Wrote", plot_path)
    except Exception as e:
        print("[warn] plot failed:", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
