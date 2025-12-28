#!/usr/bin/env python3
"""
Build a 9×9 tetrakis-square slice, delete one internal edge to
simulate a +45° deficit, then print vertex degrees.

Run:
    python lattice_sim.py          # default: centre defect
    python lattice_sim.py --none   # flat sheet (no defect)
"""

import argparse
from itertools import product

import networkx as nx


def build_sheet():
    V = [(r, c, q) for r in range(9) for c in range(9) for q in "ABCD"]
    G = nx.Graph()
    G.add_nodes_from(V)

    def clique(nodes):
        for i, v in enumerate(nodes):
            for w in nodes[i + 1 :]:
                G.add_edge(v, w)

    for r, c in product(range(9), repeat=2):
        clique([(r, c, q) for q in "ABCD"])  # intra-cell
    for r in range(9):
        for q in "ABCD":
            clique([(r, c, q) for c in range(9)])  # rows
    for c in range(9):
        for q in "ABCD":
            clique([(r, c, q) for r in range(9)])  # columns
    return G


def main(no_defect: bool):
    G = build_sheet()

    if not no_defect:
        # remove one edge => +45° wedge deficit
        G.remove_edge((4, 4, "A"), (4, 4, "B"))

    degs = G.degree((4, 4, "A")), G.degree((4, 4, "B"))
    print("Degrees at defect vertices:", *degs)
    print("Edges:", G.number_of_edges())


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument(
        "--none",
        action="store_true",
        dest="no_defect",  # <── add this line
        help="Build a flat sheet (no wedge removed).",
    )
    main(**vars(p.parse_args()))
