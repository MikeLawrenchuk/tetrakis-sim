from __future__ import annotations

import argparse
import ast
from typing import Any

from tetrakis_sim.defects import apply_defect
from tetrakis_sim.lattice import build_sheet
from tetrakis_sim.physics import run_fft, run_wave_sim
from tetrakis_sim.plot import plot_fft, plot_lattice


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tetrakis-Sim: Discrete Geometry Simulator (quick interactive CLI)"
    )

    parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--layers", type=int, default=3)

    parser.add_argument(
        "--defect",
        "--defect_type",
        dest="defect_type",
        type=str,
        default="none",
        choices=["none", "wedge", "blackhole", "singularity"],
    )
    parser.add_argument("--radius", type=float, default=None)

    parser.add_argument("--sing_mass", type=float, default=1000.0)
    parser.add_argument("--sing_potential", type=float, default=0.0)
    parser.add_argument("--sing_radius", type=float, default=None)
    parser.add_argument("--sing_prune_edges", action="store_true")

    parser.add_argument(
        "--physics",
        type=str,
        default="none",
        choices=["none", "wave", "fft"],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--damping", type=float, default=0.0)

    parser.add_argument("--kick", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_layer", type=int, default=None)

    args = parser.parse_args()

    G = build_sheet(size=args.size, dim=args.dim, layers=args.layers)

    center2d: tuple[int, int] = (args.size // 2, args.size // 2)
    center3d: tuple[int, int, int] = (args.size // 2, args.size // 2, args.layers // 2)
    center_any: tuple[int, ...] = center2d if args.dim == 2 else center3d

    removed_nodes: list[Any] = []
    if args.defect_type != "none":
        defect_kwargs: dict[str, Any] = {}

        if args.defect_type == "blackhole":
            r = args.radius if args.radius is not None else (args.size / 4)
            defect_kwargs = {"center": center_any, "radius": float(r)}

        elif args.defect_type == "wedge":
            if args.dim == 2:
                defect_kwargs = {"center": center2d}
            else:
                defect_kwargs = {"center": center2d, "layer": center3d[2]}

        elif args.defect_type == "singularity":
            sr = (
                args.sing_radius
                if args.sing_radius is not None
                else (args.radius if args.radius is not None else 0.0)
            )
            defect_kwargs = {
                "center": center_any,
                "radius": float(sr),
                "mass": float(args.sing_mass),
                "potential": float(args.sing_potential),
                "prune_edges": bool(args.sing_prune_edges),
            }

        G, removed_nodes = apply_defect(
            G,
            defect_type=args.defect_type,
            return_removed=True,
            **defect_kwargs,
        )

    def _distance_sq(node: Any) -> int:
        r, c = node[:2]
        return (r - center2d[0]) ** 2 + (c - center2d[1]) ** 2

    initial_node: Any | None = None
    if args.physics in {"wave", "fft"}:
        if args.kick:
            initial_node = ast.literal_eval(args.kick)
            if initial_node not in G:
                raise ValueError(f"--kick node {initial_node} is not in the graph")
        else:
            if args.dim == 3:
                z = center3d[2]
                nodes = [n for n in G if len(n) > 3 and n[2] == z]
            else:
                nodes = list(G.nodes)

            candidates = [
                n for n in nodes if G.degree[n] > 0 and not bool(G.nodes[n].get("singular", False))
            ]
            pool = candidates if candidates else nodes
            initial_node = min(pool, key=_distance_sq) if pool else None

    history = None
    freq = spectrum = values = None

    if args.physics in {"wave", "fft"}:
        initial_data = {initial_node: 1.0} if initial_node is not None else None
        history = run_wave_sim(
            G,
            steps=args.steps,
            initial_data=initial_data,
            c=args.c,
            dt=args.dt,
            damping=args.damping,
        )

        if args.physics == "fft" and initial_node is not None:
            freq, spectrum, values = run_fft(history, node=initial_node)

    if args.plot:
        final_state = history[-1] if history else None
        if args.dim == 3:
            layer = args.plot_layer if args.plot_layer is not None else center3d[2]
            plot_lattice(G, data=final_state, layer=layer)
        else:
            plot_lattice(G, data=final_state)

        if freq is not None and spectrum is not None:
            plot_fft(freq, spectrum, node=initial_node, values=values)

    print(
        "Done!",
        f"Nodes: {G.number_of_nodes()}",
        f"Edges: {G.number_of_edges()}",
        f"Removed nodes: {len(removed_nodes)}",
    )
    if initial_node is not None:
        print(
            "kick node:",
            initial_node,
            "degree:",
            G.degree[initial_node],
            "singular:",
            G.nodes[initial_node].get("singular", False),
        )
    if history is not None:
        print("Simulation metadata:", history.metadata)


if __name__ == "__main__":
    main()
