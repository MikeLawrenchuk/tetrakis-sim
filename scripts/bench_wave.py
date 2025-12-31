#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from tetrakis_sim.lattice import build_sheet
from tetrakis_sim.physics import run_wave_sim


@dataclass(frozen=True)
class BenchRow:
    dim: int
    layers: int
    size: int
    nodes: int
    edges: int
    max_degree: int
    build_s: float
    sim_s_median: float
    sim_s_min: float
    sim_s_max: float
    effective_dt: float
    stability_adjusted: bool


def _parse_int_list(parts: Sequence[str]) -> list[int]:
    out: list[int] = []
    for item in parts:
        for token in item.split(","):
            token = token.strip()
            if token:
                out.append(int(token))
    return out


def _kick_node(*, size: int, dim: int, layers: int):
    if dim == 2:
        return (size // 2, size // 2, "A")
    return (size // 2, size // 2, layers // 2, "A")


def bench_one(
    *,
    size: int,
    dim: int,
    layers: int,
    steps: int,
    c: float,
    dt: float,
    damping: float,
    repeats: int,
) -> BenchRow:
    t0 = time.perf_counter()
    G = build_sheet(size=size, dim=dim, layers=layers)
    build_s = time.perf_counter() - t0

    kick = _kick_node(size=size, dim=dim, layers=layers)
    initial_data = {kick: 1.0}

    sim_times: list[float] = []
    effective_dt = float(dt)
    stability_adjusted = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(int(repeats)):
            t1 = time.perf_counter()
            hist = run_wave_sim(
                G,
                steps=int(steps),
                initial_data=initial_data,
                c=float(c),
                dt=float(dt),
                damping=float(damping),
            )
            sim_times.append(time.perf_counter() - t1)

            effective_dt = float(hist.dt)
            stability_adjusted = bool(hist.metadata.get("stability_adjusted", False))

    nodes = list(G.nodes)
    max_degree = max((G.degree[n] for n in nodes), default=0)

    return BenchRow(
        dim=int(dim),
        layers=int(layers if dim == 3 else 1),
        size=int(size),
        nodes=int(G.number_of_nodes()),
        edges=int(G.number_of_edges()),
        max_degree=int(max_degree),
        build_s=float(build_s),
        sim_s_median=float(statistics.median(sim_times)),
        sim_s_min=float(min(sim_times)),
        sim_s_max=float(max(sim_times)),
        effective_dt=float(effective_dt),
        stability_adjusted=bool(stability_adjusted),
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark build_sheet + run_wave_sim scaling.")
    p.add_argument("--sizes", nargs="+", default=["11,21,31"], help="Comma-separated sizes.")
    p.add_argument("--dim", type=int, choices=[2, 3], default=2)
    p.add_argument("--layers", type=int, default=5, help="Used only when --dim 3.")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--damping", type=float, default=0.0)
    args = p.parse_args(argv)

    sizes = _parse_int_list(args.sizes)
    if not sizes:
        raise SystemExit("No sizes provided.")

    print(
        "dim,layers,size,nodes,edges,max_degree,build_s,sim_s_median,sim_s_min,sim_s_max,effective_dt,stability_adjusted"
    )
    for s in sizes:
        row = bench_one(
            size=s,
            dim=args.dim,
            layers=args.layers,
            steps=args.steps,
            c=args.c,
            dt=args.dt,
            damping=args.damping,
            repeats=args.repeats,
        )
        print(
            f"{row.dim},{row.layers},{row.size},{row.nodes},{row.edges},{row.max_degree},"
            f"{row.build_s:.6f},{row.sim_s_median:.6f},{row.sim_s_min:.6f},{row.sim_s_max:.6f},"
            f"{row.effective_dt:.6f},{int(row.stability_adjusted)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
