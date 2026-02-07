from typing import Final

import networkx as nx


class SovereignNode:
    """
    A 'Qubit' node in the Sovereign River.
    Represents an Indigenous Nation or a community pillar.
    """

    def __init__(self, name, node_type="Nation"):
        self.name = name
        self.node_type = node_type  # e.g., 'Housing', 'Banking', 'Nation'
        self.entropy = 1.0  # Starts high (fractured)
        self.resonance = 0.0  # Goal is to increase this
        self.entangled_nodes = []  # Your 'Quantum' connections


class RiverBlockTree:
    """
    The branching architecture that bypasses linear 'dam' systems.
    """

    def __init__(self, root_nation_name):
        self.graph = nx.DiGraph()  # Directed graph (the flow of the river)
        self.root = SovereignNode(root_nation_name, node_type="Root")
        self.graph.add_node(self.root.name, data=self.root)

    def add_pillar(self, pillar_name):
        """Adds one of the 4 Pillars as a branch from the root."""
        new_pillar = SovereignNode(pillar_name, node_type="Pillar")
        self.graph.add_edge(self.root.name, pillar_name)
        return new_pillar

    def calculate_flow(self):
        """
        The logic for your 'Quantum River'.
        This is where the Prime-Resonance Gate will live.
        """
        pass


def apply_prime_gate(node: SovereignNode, last_digit: int) -> None:
    """
    Tunes resonance based on the 'color' (last digit) of the prime.

    1 = High Stability (High Resonance)
    3 = Rare/Catalyst (High Shift)
    7/9 = Flow/Energy
    """
    weights: Final[dict[int, float]] = {1: 0.25, 3: 0.50, 7: 0.15, 9: 0.15}

    boost = weights.get(last_digit, 0.05)

    node.entropy -= boost
    node.resonance += boost

    print(
        f"Node {node.name} tuned by Prime Digit {last_digit}. "
        f"Resonance: {round(node.resonance, 2)}"
    )


def apply_hydro_drain(node: SovereignNode, intensity: float = 0.4) -> None:
    """
    Simulates the Hydro Dam/Extractive Policy.

    Actively increases entropy and siphons away resonance.
    """
    node.entropy += intensity
    node.resonance -= intensity
    if node.resonance < 0:
        node.resonance = 0.0
    print(
        f"!!! EXTERNAL DRAIN: Hydro Dam siphoning {node.name}. "
        f"Resonance dropped to {round(node.resonance, 2)}"
    )
