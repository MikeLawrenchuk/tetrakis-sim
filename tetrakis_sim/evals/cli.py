from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import run_nearest_centroid_baseline
from .dataset import generate_defect_classification_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tetrakis-eval",
        description="Evaluation harness for tetrakis-sim (dataset generation + baseline scoring).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate an eval dataset (JSONL)")
    g.add_argument("--out", required=True, help="Output JSONL path")
    g.add_argument("--n-per-class", type=int, default=10)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--size", type=int, default=11)
    g.add_argument("--dim", type=int, default=2, choices=[2, 3])
    g.add_argument("--layers", type=int, default=5)
    g.add_argument("--steps", type=int, default=40)
    g.add_argument("--c", type=float, default=1.0)
    g.add_argument("--dt", type=float, default=0.2)
    g.add_argument("--damping", type=float, default=0.0)

    b = sub.add_parser("baseline", help="Run a nearest-centroid baseline")
    b.add_argument("--data", required=True, help="Input JSONL path")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--test-frac", type=float, default=0.3)
    b.add_argument("--split", choices=["random", "stratified"], default="stratified")
    b.add_argument("--out-metrics", default=None, help="Optional path to write metrics as JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        out = Path(args.out)
        path = generate_defect_classification_jsonl(
            out,
            n_per_class=args.n_per_class,
            seed=args.seed,
            size=args.size,
            dim=args.dim,
            layers=args.layers,
            steps=args.steps,
            c=args.c,
            dt=args.dt,
            damping=args.damping,
        )
        print(f"wrote: {path}")
        return 0

    if args.cmd == "baseline":
        metrics = run_nearest_centroid_baseline(
            args.data,
            seed=args.seed,
            test_frac=args.test_frac,
            split=args.split,
        )
        for k in sorted(metrics):
            print(f"{k}: {metrics[k]}")

        if args.out_metrics:
            out = Path(args.out_metrics)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"wrote_metrics_json: {out}")

        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
