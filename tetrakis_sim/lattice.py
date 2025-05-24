# tetrakis_sim/lattice.py

import networkx as nx
from itertools import product

def build_sheet(size=9):
    """
    Build a tetrakis-square lattice (2D), size × size.

    Each cell has four nodes (A, B, C, D).
    Returns a networkx.Graph instance.
    """
    V = [(r, c, q) for r in range(size) for c in range(size) for q in "ABCD"]
    G = nx.Graph()
    G.add_nodes_from(V)

    def clique(nodes):
        for i, v in enumerate(nodes):
            for w in nodes[i+1:]:
                G.add_edge(v, w)

    # Intra-cell connections (each cell forms a clique of its A, B, C, D nodes)
    for r, c in product(range(size), repeat=2):
        clique([(r, c, q) for q in "ABCD"])
    # Row connections (link same letter across the row)
    for r in range(size):
        for q in "ABCD":
            clique([(r, c, q) for c in range(size)])
    # Column connections (link same letter across the column)
    for c in range(size):
        for q in "ABCD":
            clique([(r, c, q) for r in range(size)])
    return G


def build_sheet_3d(size=9, layers=3):
    # ... (future extension)
    pass
