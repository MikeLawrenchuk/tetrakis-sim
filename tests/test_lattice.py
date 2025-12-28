# tests/test_lattice.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest

from tetrakis_sim.lattice import build_sheet, build_sheet_3d


def test_build_sheet_2d():
    G = build_sheet(size=4)
    # 4x4 grid, 4 nodes per cell: 4*4*4 = 64 nodes
    assert G.number_of_nodes() == 64
    # Check that there are edges (not empty)
    assert G.number_of_edges() > 0
    # Check that a typical node exists
    assert (0, 0, "A") in G.nodes


def test_build_sheet_3d():
    G = build_sheet(size=3, dim=3, layers=2)
    # 3x3 grid, 2 layers, 4 nodes per cell: 3*3*2*4 = 72 nodes
    assert G.number_of_nodes() == 72
    # Check that there are vertical edges
    assert G.has_edge((0, 0, 0, "A"), (0, 0, 1, "A"))
    # Check that a typical node exists
    assert (2, 1, 1, "D") in G.nodes


def test_build_sheet_3d_wrapper():
    G = build_sheet_3d(size=2, layers=2)
    # 2x2 grid, 2 layers, 4 nodes per cell: 2*2*2*4 = 32 nodes
    assert G.number_of_nodes() == 32


if __name__ == "__main__":
    pytest.main([__file__])
