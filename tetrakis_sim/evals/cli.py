from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tetrakis-eval",
        description="Evaluation harness for tetrakis-sim (dataset generation + baseline scoring).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate an eval dataset (JSONL)")
    g.add_argument("--out", required=True, help="Output JSONL path")

    b = sub.add_parser("baseline", help="Run a baseline model on a dataset")
    b.add_argument("--data", required=True, help="Input JSONL path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        print("Not implemented yet (Milestone 3).")
        print(f"Requested output: {args.out}")
        return 0

    if args.cmd == "baseline":
        print("Not implemented yet (Milestone 3).")
        print(f"Requested dataset: {args.data}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
