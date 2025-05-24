import argparse
from tetrakis_sim.lattice import build_sheet  # , build_sheet_3d (for future 3D)
from tetrakis_sim.defects import apply_defect
from tetrakis_sim.physics import run_wave_sim, run_fft
from tetrakis_sim.plot import plot_lattice, plot_fft

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
    
    G = build_sheet(size=args.size)


    # Apply defect if specified
    if args.defect != "none":
        G = apply_defect(G, defect_type=args.defect)
    else:
        data = None

    # Apply physics simulation if specified
    if args.physics == "wave":
        data = run_wave_sim(G)
    elif args.physics == "fft":
        data = run_fft(G)
    else:
        data = None

    # Visualize if requested
    if args.plot:
        plot_lattice(G, data=data)
        if args.physics == "fft" and data is not None:
            plot_fft(data)

    print(f"Done! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

