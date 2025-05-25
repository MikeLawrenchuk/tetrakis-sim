# tetrakis_sim/defects.py

import math
import networkx as nx
from typing import Tuple, List, Optional

def apply_wedge_defect(G: nx.Graph, center: Optional[Tuple[int, int]] = None) -> nx.Graph:
    """
    Apply a +45° wedge deficit at the grid center.
    Removes edge between center 'A' and 'B' nodes (2D only).
    """
    if center is None:
        # Infer center from node labels (works for 2D)
        rows = [r for r, _, _ in G.nodes]
        cols = [c for _, c, _ in G.nodes]
        center = (rows[len(rows)//2], cols[len(cols)//2])
    try:
        G.remove_edge((center[0], center[1], "A"), (center[0], center[1], "B"))
    except Exception:
        print(f"No edge found at {center} to remove.")
    return G

def apply_blackhole_defect(
    G: nx.Graph,
    center: Tuple[int, ...],
    radius: float
) -> List:
    """
    Removes all nodes (and their edges) within a given radius of center.
    Returns the list of removed nodes.
    Works for both 2D and 3D lattices.
    """
    nodes_to_remove = []
    if len(center) == 2:
        r0, c0 = center
        for node in list(G.nodes):
            r, c = node[:2]
            if math.hypot(r - r0, c - c0) < radius:
                nodes_to_remove.append(node)
    else:
        r0, c0, z0 = center
        for node in list(G.nodes):
            r, c, z = node[:3]
            if math.sqrt((r - r0)**2 + (c - c0)**2 + (z - z0)**2) < radius:
                nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)
    return nodes_to_remove

def find_event_horizon(
    G: nx.Graph,
    removed_nodes: List,
    radius: float,
    center: Tuple[int, ...]
) -> List:
    """
    Finds nodes just outside the black hole (event horizon):
    These are nodes within one grid spacing of the radius and adjacent to removed nodes.
    """
    horizon = set()
    removed_set = set(removed_nodes)
    for node in G.nodes:
        # Compute distance from center
        if len(center) == 2:
            r, c = node[:2]
            dist = math.hypot(r - center[0], c - center[1])
        else:
            r, c, z = node[:3]
            dist = math.sqrt((r - center[0])**2 + (c - center[1])**2 + (z - center[2])**2)
        # "Shell" just outside radius, and must touch a removed node
        if radius - 1 <= dist < radius + 1:
            if any(neigh in removed_set for neigh in G.neighbors(node)):
                horizon.add(node)
    return list(horizon)

def apply_defect(
    G: nx.Graph,
    defect_type: str = "wedge",
    **kwargs
) -> nx.Graph:
    """
    Dispatches to the appropriate defect function.
    """
    if defect_type == "wedge":
        return apply_wedge_defect(G, **kwargs)
    elif defect_type == "blackhole":
        apply_blackhole_defect(G, **kwargs)
        return G
    # Add more defect types here in the future
    return G
