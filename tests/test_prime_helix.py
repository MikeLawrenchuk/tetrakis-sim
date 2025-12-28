import networkx as nx

from tetrakis_sim.prime_helix import add_prime_helix


def test_ring_and_node_counts():
    G = nx.Graph()
    add_prime_helix(G, n_rings=6)
    assert {d["ring"] for _, d in G.nodes(data=True)} == set(range(6))
    # each diamond has 5 vertices
    assert len(G) == 6 * 5


def test_positions_have_three_coords():
    G = nx.Graph()
    add_prime_helix(G, n_rings=1)
    n, d = next(iter(G.nodes(data=True)))
    assert len(d["pos"]) == 3
