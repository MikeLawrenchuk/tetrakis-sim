# tetrakis_sim/lattice.py

from itertools import product

import networkx as nx


def clique(G, nodes):
    """Helper to connect all nodes in a clique."""
    for i, v in enumerate(nodes):
        for w in nodes[i + 1 :]:
            G.add_edge(v, w)


def build_sheet(size=9, dim=2, layers=3):
    """
    Build a tetrakis-square lattice.

    - 2D: dim=2 (default), returns G with nodes (r, c, q)
    - 3D: dim=3, returns G with nodes (r, c, z, q), with vertical edges between floors

    Args:
        size (int): grid size per dimension
        dim (int): 2 (default) or 3
        layers (int): number of layers for 3D

    Returns:
        networkx.Graph
    """
    if dim == 2:
        V = [(r, c, q) for r in range(size) for c in range(size) for q in "ABCD"]
        G = nx.Graph()
        G.add_nodes_from(V)
        # Intra-cell clique
        for r, c in product(range(size), repeat=2):
            clique(G, [(r, c, q) for q in "ABCD"])
        # Row
        for r in range(size):
            for q in "ABCD":
                clique(G, [(r, c, q) for c in range(size)])
        # Column
        for c in range(size):
            for q in "ABCD":
                clique(G, [(r, c, q) for r in range(size)])
        return G

    elif dim == 3:
        V = [
            (r, c, z, q)
            for r in range(size)
            for c in range(size)
            for z in range(layers)
            for q in "ABCD"
        ]
        G = nx.Graph()
        G.add_nodes_from(V)
        # Intra-cell, row, col in each floor
        for z in range(layers):
            for r, c in product(range(size), repeat=2):
                clique(G, [(r, c, z, q) for q in "ABCD"])
            for r in range(size):
                for q in "ABCD":
                    clique(G, [(r, c, z, q) for c in range(size)])
            for c in range(size):
                for q in "ABCD":
                    clique(G, [(r, c, z, q) for r in range(size)])
        # Vertical (between floors)
        for z in range(layers - 1):
            for r in range(size):
                for c in range(size):
                    for q in "ABCD":
                        G.add_edge((r, c, z, q), (r, c, z + 1, q))
        return G

    else:
        raise ValueError("dim must be 2 or 3")


# For backward compatibility with your original code
def build_sheet_3d(size=9, layers=3):
    """Build a 3D tetrakis-square lattice (wrapper for build_sheet)."""
    return build_sheet(size=size, dim=3, layers=layers)
