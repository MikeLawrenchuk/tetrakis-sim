# scripts/run_batch.py

import argparse
import os
import numpy as np
import json
from datetime import datetime
from tetrakis_sim.lattice import build_sheet
from tetrakis_sim.defects import apply_blackhole_defect
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
    parser.add_argument('--layers', type=int, default=5, help='Number of lattice layers (z)')
    parser.add_argument('--radius', type=float, default=2.5, help='Black hole radius')
    parser.add_argument('--steps', type=int, default=40, help='Number of simulation time steps')
    parser.add_argument('--c', type=float, default=1.0, help='Wave speed (default=1.0)')
    parser.add_argument('--dt', type=float, default=0.2, help='Time step size (default=0.2)')
    parser.add_argument('--damping', type=float, default=0.0, help='Wave damping (default=0.0)')
    parser.add_argument('--defect_type', type=str, default='blackhole', choices=['blackhole', 'wedge', 'none'], help='Type of defect to apply')
    parser.add_argument('--outdir', type=str, default='batch_cli_output', help='Directory to save output')
    parser.add_argument('--prefix', type=str, default=None, help='Custom prefix for output files (optional)')

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix or f"size{args.size}_radius{args.radius}_layers{args.layers}_steps{args.steps}"

    print(f"== Tetrakis-Sim Batch CLI ==")
    print(f"Parameters: {vars(args)}")

    # Build lattice and apply defect
    G = build_sheet(size=args.size, dim=3, layers=args.layers)
    bh_center = (args.size//2, args.size//2, args.layers//2)
    removed_nodes = apply_blackhole_defect(G, bh_center, args.radius)
    z = args.layers // 2
    nodes_on_layer = [n for n in G if n[2] == z]
    initial_node = nodes_on_layer[len(nodes_on_layer)//2]
    initial_data = {initial_node: 1.0}

    history = run_wave_sim(
        G, steps=args.steps, initial_data=initial_data,
        c=args.c, dt=args.dt, damping=args.damping
    )

    # FFT analysis and plot
    freq, spectrum, values = run_fft(history, initial_node)
    plot_filename = os.path.join(args.outdir, f"{prefix}_fft_node{initial_node}.png")
    import matplotlib.pyplot as plt
    plot_fft(freq, spectrum, node=initial_node, values=values)
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved FFT plot to {plot_filename}")

    # Save spectrum CSV
    csv_filename = os.path.join(args.outdir, f"{prefix}_spectrum.csv")
    np.savetxt(csv_filename, np.column_stack([freq, spectrum]), delimiter=',', header='freq,spectrum')
    print(f"Saved FFT data to {csv_filename}")

    # Save run metadata as JSON
    metadata = vars(args).copy()
    metadata['kick_node'] = str(initial_node)
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
