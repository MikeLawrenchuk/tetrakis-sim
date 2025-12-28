"""
examples/ulam_diagonal_helix_demo.py

Demo: build a prime-helix whose ring radii are selected using an Ulam-spiral rule.

- We compute primes among integers 1..N laid out on an Ulam spiral.
- We select primes that lie on the two main diagonals (x=y or x=-y).
- We feed those primes as radii into add_prime_helix(..., radii=...).

Run:
  python examples/ulam_diagonal_helix_demo.py
"""

from __future__ import annotations

import math
import networkx as nx

from tetrakis_sim.prime_helix import add_prime_helix
from tetrakis_sim.ulam import primes_on_ulam_diagonals

# If your project exposes a 3D plot helper, use it.
# If this import fails, comment it out and just print node/edge counts.
from tetrakis_sim.plot import plot_3d_graph


def main() -> None:
    # Ulam selection size: primes among integers 1..ULAM_N
    ULAM_N = 50_000

    # How many rings to build from the selected prime list
    N_RINGS = 150

    # Helix geometry controls
    DTHETA_DEG = 6.0
    PITCH = 1.8

    radii = primes_on_ulam_diagonals(ULAM_N)

    if len(radii) == 0:
        raise RuntimeError(f"No diagonal primes found up to {ULAM_N} (unexpected).")

    G = nx.Graph()
    add_prime_helix(
        G,
        n_rings=N_RINGS,
        radii=radii,
        dtheta=math.radians(DTHETA_DEG),
        pitch=PITCH,
    )

    print(
        f"Built prime-helix with {N_RINGS} rings from Ulam diagonal primes <= {ULAM_N}.\n"
        f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )

    # Visualize (if available)
    plot_3d_graph(G, node_size=10, title="Prime-Helix (Ulam diagonal prime radii)")


if __name__ == "__main__":
    main()

