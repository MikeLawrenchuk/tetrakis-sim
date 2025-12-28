# tetrakis_sim/defects.py

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, cast

import networkx as nx


@dataclass(frozen=True)
class DefectResult:
    """Container describing the side-effects of a defect application.

    Attributes
    ----------
    graph:
        The mutated graph (identical object passed into the defect helper).
    removed_nodes:
        Sequence of nodes removed from the graph.  Only populated for defects
        that physically delete nodes (e.g., the black-hole defect).
    removed_edges:
        Sequence of edges removed from the graph.  Used for defects such as the
        wedge deficit which only excises edges.
    metadata:
        Free-form dictionary for future extensions.
    """

    graph: nx.Graph
    removed_nodes: tuple[Any, ...] = ()
    removed_edges: tuple[Any, ...] = ()
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def __iter__(self):  # pragma: no cover - convenience for unpacking
        yield self.graph
        yield list(self.removed_nodes)


def _infer_center_coordinates(G: nx.Graph) -> tuple[int, int]:
    rows = sorted({node[0] for node in G.nodes})
    cols = sorted({node[1] for node in G.nodes})
    return rows[len(rows) // 2], cols[len(cols) // 2]


def _infer_layers(G: nx.Graph) -> list[int]:
    return sorted({node[2] for node in G.nodes if len(node) > 3})


def apply_wedge_defect(
    G: nx.Graph,
    center: tuple[int, int] | None = None,
    layer: int | None = None,
    *,
    return_result: bool = False,
) -> nx.Graph | DefectResult:
    """Apply a +45° wedge deficit around the grid centre."""

    if not G.nodes:
        result = DefectResult(G)
        return result if return_result else result.graph

    node_length = len(next(iter(G.nodes)))
    if node_length not in (3, 4):
        raise ValueError("Unsupported node format for wedge defect")

    if center is None:
        center = _infer_center_coordinates(G)

    removed_edges: list[Any] = []

    if node_length == 3:
        edge2d = ((center[0], center[1], "A"), (center[0], center[1], "B"))
        if G.has_edge(*edge2d):
            G.remove_edge(*edge2d)
            removed_edges.append(edge2d)
        else:  # pragma: no cover
            warnings.warn(f"No wedge edge found at {center} to remove.")
        result = DefectResult(G, removed_edges=tuple(removed_edges))
        return result if return_result else result.graph

    layers = _infer_layers(G)
    if not layers:
        raise ValueError("Expected z-layer component for 3-D wedge defect")

    if layer is None:
        layer = layers[len(layers) // 2]

    if layer not in layers:
        raise ValueError(f"Layer {layer} is outside of lattice range {layers}")

    edge3d = (
        (center[0], center[1], layer, "A"),
        (center[0], center[1], layer, "B"),
    )
    if G.has_edge(*edge3d):
        G.remove_edge(*edge3d)
        removed_edges.append(edge3d)
    else:  # pragma: no cover
        warnings.warn(f"No wedge edge found at {center} on layer {layer} to remove.")
    result = DefectResult(G, removed_edges=tuple(removed_edges))
    return result if return_result else result.graph


def apply_blackhole_defect(
    G: nx.Graph,
    center: tuple[int, ...],
    radius: float,
    *,
    return_result: bool = False,
) -> list[Any] | DefectResult:
    """Remove all nodes within ``radius`` of ``center``."""

    nodes_to_remove: list[Any] = []
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
            if math.sqrt((r - r0) ** 2 + (c - c0) ** 2 + (z - z0) ** 2) < radius:
                nodes_to_remove.append(node)

    G.remove_nodes_from(nodes_to_remove)
    result = DefectResult(G, removed_nodes=tuple(nodes_to_remove))
    return result if return_result else list(result.removed_nodes)


def find_event_horizon(
    G: nx.Graph,
    removed_nodes: list[Any],
    radius: float,
    center: tuple[int, ...],
    *,
    adjacency_graph: nx.Graph | None = None,
) -> list[Any]:
    """Find nodes just outside the black hole (event horizon)."""

    horizon = set()
    removed_set = set(removed_nodes)
    adjG = adjacency_graph or G

    for node in G.nodes:
        if len(center) == 2:
            r, c = node[:2]
            dist = math.hypot(r - center[0], c - center[1])
        else:
            r, c, z = node[:3]
            dist = math.sqrt((r - center[0]) ** 2 + (c - center[1]) ** 2 + (z - center[2]) ** 2)

        if radius - 1 <= dist < radius + 1:
            if any(neigh in removed_set for neigh in adjG.neighbors(node)):
                horizon.add(node)

    return list(horizon)


def apply_defect(
    G: nx.Graph,
    defect_type: str = "wedge",
    *,
    return_removed: bool = False,
    **kwargs,
) -> nx.Graph | tuple[nx.Graph, list[Any]]:
    """Dispatch onto the requested defect helper."""

    result: DefectResult

    if defect_type == "none":
        result = DefectResult(G)
    elif defect_type == "wedge":
        result = cast(DefectResult, apply_wedge_defect(G, return_result=True, **kwargs))
    elif defect_type == "blackhole":
        result = cast(DefectResult, apply_blackhole_defect(G, return_result=True, **kwargs))
    elif defect_type == "singularity":
        result = cast(DefectResult, apply_singularity_defect(G, return_result=True, **kwargs))
    else:
        raise ValueError(f"Unsupported defect type '{defect_type}'")

    if return_removed:
        return result.graph, list(result.removed_nodes)
    return result.graph


def apply_singularity_defect(
    G: nx.Graph,
    center: tuple[int, ...],
    *,
    mass: float = 1000.0,
    potential: float = 0.0,
    radius: float = 0.0,
    prune_edges: bool = False,
    return_result: bool = False,
) -> nx.Graph | DefectResult:
    """Naked singularity defect: tag nodes near the center with mass/potential."""

    if not G.nodes:
        result = DefectResult(G)
        return result if return_result else result.graph

    tagged_nodes: list[Any] = []
    removed_edges: list[Any] = []

    for node in list(G.nodes):
        if len(center) == 2:
            r, c = node[:2]
            dist = math.hypot(r - center[0], c - center[1])
        else:
            r, c, z = node[:3]
            dist = math.sqrt((r - center[0]) ** 2 + (c - center[1]) ** 2 + (z - center[2]) ** 2)

        if dist <= radius:
            tagged_nodes.append(node)
            G.nodes[node]["mass"] = float(mass)
            G.nodes[node]["potential"] = float(potential)
            G.nodes[node]["singular"] = True

    if prune_edges:
        for n in tagged_nodes:
            for nb in list(G.neighbors(n)):
                if G.has_edge(n, nb):
                    G.remove_edge(n, nb)
                    removed_edges.append((n, nb))

    result = DefectResult(
        G,
        removed_nodes=tuple(),
        removed_edges=tuple(removed_edges),
        metadata={
            "singularity_center": center,
            "singularity_radius": radius,
            "singularity_mass": mass,
            "singularity_potential": potential,
            "singularity_prune_edges": prune_edges,
            "singularity_tagged_count": len(tagged_nodes),
        },
    )
    return result if return_result else result.graph
