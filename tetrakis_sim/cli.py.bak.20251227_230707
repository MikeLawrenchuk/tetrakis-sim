import argparse
import ast

from tetrakis_sim.defects import apply_defect
from tetrakis_sim.lattice import build_sheet
from tetrakis_sim.physics import run_fft, run_wave_sim
from tetrakis_sim.plot import plot_fft, plot_lattice


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tetrakis-Sim: Discrete Geometry Simulator (quick interactive CLI)"
    )

    parser.add_argument(
        "--dim", type=int, default=2, choices=[2, 3], help="Dimension (2 or 3)"
    )
    parser.add_argument("--size", type=int, default=30, help="Grid size (NxN or NxNxN)")
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="Number of layers for 3D (used only if --dim 3)",
    )

    # Defects
    parser.add_argument(
        "--defect",
        "--defect_type",
        dest="defect_type",
        type=str,
        default="none",
        choices=["none", "wedge", "blackhole", "singularity"],
        help="Defect type (none, wedge, blackhole, singularity)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Radius for blackhole (and default for singularity if --sing_radius not set)",
    )

    # Singularity parameters
    parser.add_argument(
        "--sing_mass", type=float, default=1000.0, help="Singularity mass"
    )
    parser.add_argument(
        "--sing_potential", type=float, default=0.0, help="Singularity potential"
    )
    parser.add_argument(
        "--sing_radius",
        type=float,
        default=None,
        help="Singularity radius (default: use --radius; if both unset: 0.0)",
    )
    parser.add_argument(
        "--sing_prune_edges",
        action="store_true",
        help="Prune edges for nodes within singularity radius",
    )

    # Physics
    parser.add_argument(
        "--physics",
        type=str,
        default="none",
        choices=["none", "wave", "fft"],
        help="Physics model (none, wave, fft)",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of simulation steps"
    )
    parser.add_argument("--c", type=float, default=1.0, help="Wave speed")
    parser.add_argument("--dt", type=float, default=0.2, help="Time step size")
    parser.add_argument("--damping", type=float, default=0.0, help="Damping")

    # Kick + plotting
    parser.add_argument(
        "--kick",
        type=str,
        default=None,
        help="Kick node tuple as a Python literal, e.g. \"(10,10,'A')\" or \"(10,10,2,'A')\"",
    )
    parser.add_argument("--plot", action="store_true", help="Visualize output")
    parser.add_argument(
        "--plot_layer",
        type=int,
        default=None,
        help="For 3D plotting, which z-slice to render (default: center layer)",
    )

    args = parser.parse_args()

    # Build lattice
    G = build_sheet(size=args.size, dim=args.dim, layers=args.layers)

    # Choose geometric center
    if args.dim == 2:
        center = (args.size // 2, args.size // 2)
    else:
        center = (args.size // 2, args.size // 2, args.layers // 2)

    # Apply defect
    removed_nodes = []
    if args.defect_type != "none":
        defect_kwargs = {}

        if args.defect_type == "blackhole":
            r = args.radius if args.radius is not None else (args.size / 4)
            defect_kwargs = {"center": center, "radius": float(r)}

        elif args.defect_type == "wedge":
            if args.dim == 2:
                defect_kwargs = {"center": center}
            else:
                defect_kwargs = {"center": center[:2], "layer": center[2]}

        elif args.defect_type == "singularity":
            sr = (
                args.sing_radius
                if args.sing_radius is not None
                else (args.radius if args.radius is not None else 0.0)
            )
            defect_kwargs = {
                "center": center,
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

    # Pick kick node (default: closest-to-center, connected, non-singular)
    def _distance_sq(node):
        r, c = node[:2]
        return (r - center[0]) ** 2 + (c - center[1]) ** 2

    initial_node = None
    if args.physics in {"wave", "fft"}:
        if args.kick:
            initial_node = ast.literal_eval(args.kick)
            if initial_node not in G:
                raise ValueError(f"--kick node {initial_node} is not in the graph")
        else:
            if args.dim == 3:
                z = center[2]
                nodes = [n for n in G if len(n) > 2 and n[2] == z]
            else:
                nodes = list(G.nodes)

            candidates = [
                n
                for n in nodes
                if G.degree[n] > 0 and not G.nodes[n].get("singular", False)
            ]
            pool = candidates if candidates else nodes
            initial_node = min(pool, key=_distance_sq)

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

        if args.physics == "fft":
            freq, spectrum, values = run_fft(history, node=initial_node)

    if args.plot:
        final_state = history[-1] if history else None
        if args.dim == 3:
            layer = args.plot_layer if args.plot_layer is not None else center[2]
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
