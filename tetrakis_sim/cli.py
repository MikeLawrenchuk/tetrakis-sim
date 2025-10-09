import argparse
from tetrakis_sim.lattice import build_sheet  # , build_sheet_3d (for future 3D)
from tetrakis_sim.defects import apply_defect
from tetrakis_sim.physics import run_fft, run_wave_sim
from tetrakis_sim.plot import plot_fft, plot_lattice

def main():
    parser = argparse.ArgumentParser(description="Tetrakis-Sim: Discrete Geometry Simulator")
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3], help="Dimension of lattice (2 or 3)")
    parser.add_argument("--size", type=int, default=30, help="Grid size (NxN or NxNxN)")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers for 3D grid (only used if --dim 3)")
    parser.add_argument("--defect", type=str, default="none", help="Defect type (none, wedge, blackhole, custom)")
    parser.add_argument("--physics", type=str, default="none", help="Physics model (none, wave, fft)")
    parser.add_argument("--plot", action="store_true", help="Visualize output")
    args = parser.parse_args()

    # Build the lattice (2D or 3D)
    # if args.dim == 3:
    #    G = build_sheet_3d(size=args.size, layers=args.layers)
    # else:
    #    G = build_sheet(size=args.size)
    
    G = build_sheet(size=args.size, dim=args.dim, layers=args.layers)

    removed_nodes = []
    if args.defect != "none":
        center_rc = (args.size // 2, args.size // 2)
        defect_kwargs = {}
        if args.defect == "blackhole":
            if args.dim == 3:
                center = (center_rc[0], center_rc[1], args.layers // 2)
            else:
                center = center_rc
            defect_kwargs = {"center": center, "radius": args.size / 4}
        elif args.defect == "wedge":
            defect_kwargs = {"center": center_rc}
            if args.dim == 3:
                defect_kwargs["layer"] = args.layers // 2
        G, removed_nodes = apply_defect(
            G,
            defect_type=args.defect,
            return_removed=True,
            **defect_kwargs,
        )

    history = None
    fft_payload = None
    if args.physics in {"wave", "fft"}:
        history = run_wave_sim(G)
        if args.physics == "fft":
            freq, spectrum, values = run_fft(history)
            fft_payload = (freq, spectrum, values)

    if args.plot:
        final_state = history[-1] if history else None
        plot_lattice(G, data=final_state)
        if fft_payload is not None:
            plot_fft(*fft_payload)

    print(
        "Done!",
        f"Nodes: {G.number_of_nodes()}",
        f"Edges: {G.number_of_edges()}",
        f"Removed nodes: {len(removed_nodes)}",
    )

