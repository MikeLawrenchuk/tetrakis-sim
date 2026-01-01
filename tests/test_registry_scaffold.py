from __future__ import annotations

import networkx as nx
import pytest

from tetrakis_sim.registry import get, list_names, register_defect, register_lattice


def test_registry_register_and_get() -> None:
    def lattice_fn(size: int = 3) -> nx.Graph:
        return nx.path_graph(size)

    def defect_fn(G: nx.Graph, **kwargs):
        return G, []

    register_lattice("test_lattice", lattice_fn, description="test lattice", overwrite=True)
    register_defect("test_defect", defect_fn, description="test defect", overwrite=True)

    fnL = get("lattice", "test_lattice")
    fnD = get("defect", "test_defect")

    G = fnL(4)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 4

    G2, removed = fnD(G)
    assert G2.number_of_nodes() == 4
    assert removed == []


def test_registry_missing_raises() -> None:
    with pytest.raises(KeyError):
        _ = get("lattice", "does_not_exist")


def test_list_names_includes_registered() -> None:
    # Use overwrite=True to avoid conflicts across test runs.
    register_lattice("test_list", lambda size=2: nx.path_graph(size), overwrite=True)
    assert "test_list" in list_names("lattice")
