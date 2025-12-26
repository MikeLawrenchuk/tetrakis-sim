# tetrakis_sim/batch.py

import ast
import argparse
import os
import json
from datetime import datetime

import numpy as np


from tetrakis_sim.lattice import build_sheet
from tetrakis_sim.defects import apply_defect, find_event_horizon
from tetrakis_sim.physics import run_wave_sim, run_fft
from tetrakis_sim.plot import plot_fft

def main():
    parser = argparse.ArgumentParser(
        description="""
Run a batch lattice wave simulation with custom parameters.
Example:
  python scripts/run_batch.py --size 11 --radius 2.5 --layers 7 --steps 60 --c 1.5 --dt 0.2 --damping 0.01
Outputs a plot (PNG), spectrum CSV, and metadata JSON to the output directory.
"""
    )
    parser.add_argument('--size', type=int, default=9, help='Lattice width/height (NxN)')
    parser.add_argument('--dim', type=int, default=3, choices=[2, 3], help='Lattice dimension (2 or 3)')
    parser.add_argument('--layers', type=int, default=5, help='Number of lattice layers (z)')
    parser.add_argument('--radius', type=float, default=2.5, help='Black hole radius')
    parser.add_argument('--steps', type=int, default=40, help='Number of simulation time steps')
    parser.add_argument('--c', type=float, default=1.0, help='Wave speed (default=1.0)')
    parser.add_argument('--dt', type=float, default=0.2, help='Time step size (default=0.2)')
    parser.add_argument('--damping', type=float, default=0.0, help='Wave damping (default=0.0)')
    parser.add_argument('--defect_type', type=str, default='blackhole', choices=['blackhole', 'wedge', 'none', 'singularity' ], help='Type of defect  to apply')
    parser.add_argument('--outdir', type=str, default='batch_cli_output', help='Directory to save output')
    parser.add_argument('--prefix', type=str, default=None, help='Custom prefix for output files (optional)')
    parser.add_argument( "--kick", type=str, default=None, help="Kick node tuple as a Python literal, e.g. '(10,10,\"A\")' or '(10,10,2,\"A\")'",)
    parser.add_argument('--sing_mass', type=float, default=1000.0, help='Singularity mass')
    parser.add_argument('--sing_potential', type=float, default=0.0, help='Singularity potential')
    parser.add_argument('--sing_radius', type=float, default=None, help='Singularity radius (default: use --radius)')
    parser.add_argument('--sing_prune_edges', action='store_true', help='Prune edges at singularity region')
    


    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix or f"size{args.size}_radius{args.radius}_layers{args.layers}_steps{args.steps}"

    print(f"== Tetrakis-Sim Batch CLI ==")
    print(f"Parameters: {vars(args)}")

    # Build lattice and apply defect
    G = build_sheet(size=args.size, dim=args.dim, layers=args.layers)
    G_pre_defect = G.copy()
    if args.dim == 2:
        center = (args.size // 2, args.size // 2)
    else:
        center = (args.size // 2, args.size // 2, args.layers // 2)
    defect_kwargs = {}
    if args.defect_type == "blackhole":
        defect_kwargs = {"center": center, "radius": args.radius}
    elif args.defect_type == "wedge":
        if args.dim == 2:
            defect_kwargs = {"center": center}
        else:
            defect_kwargs = {"center": center[:2], "layer": center[2]}

    elif args.defect_type == "singularity":
        sr = args.radius if args.sing_radius is None else args.sing_radius
        defect_kwargs = {
            "center": center,
            "mass": args.sing_mass,
            "potential": args.sing_potential,
            "radius": sr,
            "prune_edges": args.sing_prune_edges,
        }




    G, removed_nodes = apply_defect(
        G, defect_type=args.defect_type, return_removed=True, **defect_kwargs
    )

    if args.defect_type == "blackhole":
        horizon_nodes = find_event_horizon(G, removed_nodes, args.radius, center, adjacency_graph=G_pre_defect)

    else:
        horizon_nodes = []

    if args.dim == 3:
        z = center[2]
        nodes_on_layer = [n for n in G if len(n) > 3 and n[2] == z]
        if not nodes_on_layer:
            raise RuntimeError("No nodes remain on the central layer after defect application")
    else:
        nodes_on_layer = list(G.nodes)
        if not nodes_on_layer:
            raise RuntimeError("No nodes remain after defect application")

    # Kick the node closest to the geometric centre that survived the defect
    def _distance_sq(node):
        r, c = node[:2]
        return (r - center[0]) ** 2 + (c - center[1]) ** 2

    if args.kick:
        initial_node = ast.literal_eval(args.kick)
        if initial_node not in G:
            raise ValueError(f"--kick node {initial_node} is not in the graph after defect application")
    else:
        # Prefer connected, non-singular nodes (avoids degree-0 ramps and avoids kicking the singular point)
        candidates = [
            n for n in nodes_on_layer
            if G.degree[n] > 0 and not G.nodes[n].get("singular", False)
        ]
        pool = candidates if candidates else nodes_on_layer
        initial_node = min(pool, key=_distance_sq)

    print(
        "kick node:", initial_node,
        "degree:", G.degree[initial_node],
        "singular:", G.nodes[initial_node].get("singular", False),
    )
    initial_data = {initial_node: 1.0}





    history = run_wave_sim(
        G, steps=args.steps, initial_data=initial_data,
        c=args.c, dt=args.dt, damping=args.damping
    )

    # FFT analysis and plot
    freq, spectrum, values = run_fft(history, initial_node)
    plot_filename = os.path.join(args.outdir, f"{prefix}_fft_node{initial_node}.png")

    # plot_fft now saves the figure itself; do not call plt.savefig() here
    plot_fft(freq, spectrum, node=initial_node, values=values, save=plot_filename, show=False)

    print(f"Saved FFT plot to {plot_filename}")


    # Save spectrum CSV
    csv_filename = os.path.join(args.outdir, f"{prefix}_spectrum.csv")
    np.savetxt(csv_filename, np.column_stack([freq, spectrum]), delimiter=',', header='freq,spectrum')
    print(f"Saved FFT data to {csv_filename}")

    # Save run metadata as JSON
    metadata = vars(args).copy()
    metadata['kick_node'] = initial_node
    metadata['effective_dt'] = history.dt
    metadata['stability_adjusted'] = history.metadata.get('stability_adjusted', False)
    metadata['stability_limit'] = history.metadata.get('stability_limit')
    metadata['removed_node_count'] = len(removed_nodes)
    if removed_nodes:
        metadata['removed_nodes'] = removed_nodes
    if horizon_nodes:
        metadata['event_horizon_nodes'] = horizon_nodes
    metadata['csv_filename'] = csv_filename
    metadata['plot_filename'] = plot_filename
    metadata['run_id'] = run_id
    metadata['dominant_freq'] = float(freq[np.argmax(spectrum)])
    json_filename = os.path.join(args.outdir, f"{prefix}_metadata.json")
    with open(json_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata to {json_filename}")

if __name__ == "__main__":
    main()
