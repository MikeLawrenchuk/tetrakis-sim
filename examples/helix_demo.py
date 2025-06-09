# examples/helix_demo.py
"""
Prime-helix visual sanity-check.

Run with:
    python examples/helix_demo.py
"""

import math
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                         # noqa: F401


from tetrakis_sim.prime_helix import add_prime_helix

# ---------------------------------------------------------------------------
# Parameters you may want to tweak quickly
# ---------------------------------------------------------------------------
N_RINGS  = 12                    # how many prime rings to draw
DTHETA   = math.radians(8)       # rotation increment (° → rad)
PITCH    = 2.0                   # vertical step between rings
NODE_SZ  = 18                    # marker size in points
C_MAP = plt.get_cmap("viridis", N_RINGS)    # discrete colours per ring
# ---------------------------------------------------------------------------


def plot_3d_graph_with_edges(G: nx.Graph) -> None:
    """Scatter + thin edges for every ring to reveal diamond outlines."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # iterate rings so colour can be assigned per ring
    for k in range(N_RINGS):
        verts = [G.nodes[(k, j)]["pos"] for j in range(5)]       # 5 vertices
        xs, ys, zs = zip(*verts)

        # scatter vertices
        ax.scatter(xs, ys, zs, s=NODE_SZ, color=C_MAP(k), depthshade=True, picker=True, label=f"ring {k}",  )

        # draw outline
        ax.plot(xs, ys, zs, linewidth=0.6, color=C_MAP(k), alpha=0.7, antialiased=True, )

    ax.set_title("Prime-helix demo")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()

    fig.savefig("prime_helix_demo.png", dpi=180, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def main() -> None:
    G = nx.Graph()
    add_prime_helix(G, n_rings=N_RINGS, dtheta=DTHETA, pitch=PITCH)
    plot_3d_graph_with_edges(G)


if __name__ == "__main__":
    main()
