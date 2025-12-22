# tetrakis_sim/defects.py

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    removed_nodes: Tuple = ()
    removed_edges: Tuple = ()
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def __iter__(self):  # pragma: no cover - convenience for unpacking
        yield self.graph
        yield list(self.removed_nodes)

def _infer_center_coordinates(G: nx.Graph) -> Tuple[int, int]:
    rows = sorted({node[0] for node in G.nodes})
    cols = sorted({node[1] for node in G.nodes})
    return rows[len(rows) // 2], cols[len(cols) // 2]


def _infer_layers(G: nx.Graph) -> List[int]:
    return sorted({node[2] for node in G.nodes if len(node) > 3})


def apply_wedge_defect(
    G: nx.Graph,
    center: Optional[Tuple[int, int]] = None,
    layer: Optional[int] = None,
    *,
    return_result: bool = False,
) -> nx.Graph | DefectResult:
    """Apply a +45° wedge deficit around the grid centre.

    The classic wedge defect removes the edge between the "A" and "B" vertices
    of the central cell.  Historically the implementation only worked for 2-D
    lattices (triplets).  This helper now supports both 2-D and 3-D lattices:

    * 2-D: remove the single (A,B) edge at the requested ``center``.
    * 3-D: remove the (A,B) edge on the specified ``layer``.  If no layer is
      supplied the central layer is used.  All other layers are left intact.
    """

    if not G.nodes:
        return DefectResult(G)

    node_length = len(next(iter(G.nodes)))
    if node_length not in (3, 4):
        raise ValueError("Unsupported node format for wedge defect")

    if center is None:
        center = _infer_center_coordinates(G)

    removed_edges: List[Tuple] = []

    if node_length == 3:
        edge = ((center[0], center[1], "A"), (center[0], center[1], "B"))
        if G.has_edge(*edge):
            G.remove_edge(*edge)
            removed_edges.append(edge)
        else:  # pragma: no cover - defensive fallback
            warnings.warn(f"No wedge edge found at {center} to remove.")
        result = DefectResult(G, removed_edges=tuple(removed_edges))
        return result if return_result else result.graph

    # 3-D lattice handling
    layers = _infer_layers(G)
    if not layers:
        raise ValueError("Expected z-layer component for 3-D wedge defect")

    if layer is None:
        layer = layers[len(layers) // 2]

    if layer not in layers:
        raise ValueError(f"Layer {layer} is outside of lattice range {layers}")

    edge = (
        (center[0], center[1], layer, "A"),
        (center[0], center[1], layer, "B"),
    )
    if G.has_edge(*edge):
        G.remove_edge(*edge)
        removed_edges.append(edge)
    else:  # pragma: no cover - defensive fallback
        warnings.warn(
            f"No wedge edge found at {center} on layer {layer} to remove."
        )
    result = DefectResult(G, removed_edges=tuple(removed_edges))
    return result if return_result else result.graph

def apply_blackhole_defect(
    G: nx.Graph,
    center: Tuple[int, ...],
    radius: float,
    *,
    return_result: bool = False,
) -> List | DefectResult:
    """Remove all nodes within ``radius`` of ``center``.

    The return value retains the previous behaviour of returning the list of
    removed nodes while also bundling the mutated graph and providing a uniform
    :class:`DefectResult` container that other defects now use as well.
    """

    nodes_to_remove: List[Tuple] = []
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
    removed_nodes: List,
    radius: float,
    center: Tuple[int, ...],
    *,
    adjacency_graph: Optional[nx.Graph] = None,
) -> List:
    """
    Finds nodes just outside the black hole (event horizon):
    nodes within one grid spacing of the radius AND adjacent to removed nodes.

    Important:
    If the black-hole defect removed nodes from G, those removed nodes will no
    longer appear in G.neighbors(...). In that case, pass the *pre-defect*
    graph as adjacency_graph so adjacency to removed nodes can be evaluated.
    """
    horizon = set()
    removed_set = set(removed_nodes)
    adjG = adjacency_graph or G

    for node in G.nodes:
        # Compute distance from center
        if len(center) == 2:
            r, c = node[:2]
            dist = math.hypot(r - center[0], c - center[1])
        else:
            r, c, z = node[:3]
            dist = math.sqrt(
                (r - center[0]) ** 2
                + (c - center[1]) ** 2
                + (z - center[2]) ** 2
            )

        # "Shell" just outside radius, and must touch a removed node
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
) -> nx.Graph | Tuple[nx.Graph, List]:
    """Dispatch onto the requested defect helper.

    Parameters
    ----------
    G:
        Graph to mutate in-place.
    defect_type:
        One of ``"wedge"``, ``"blackhole"`` or ``"none"``.
    return_removed:
        If ``True`` the function returns ``(graph, removed_nodes)`` for
        backwards compatibility with legacy call-sites that only required the
        graph.  The more expressive :class:`DefectResult` is always returned to
        callers using :func:`apply_wedge_defect` or :func:`apply_blackhole_defect`
        directly.
    **kwargs:
        Forwarded to the specific defect helper.
    """

    if defect_type == "none":
        result = DefectResult(G)
    elif defect_type == "wedge":
        result = apply_wedge_defect(G, return_result=True, **kwargs)
    elif defect_type == "blackhole":
        result = apply_blackhole_defect(G, return_result=True, **kwargs)
    else:
        raise ValueError(f"Unsupported defect type '{defect_type}'")

    if return_removed:
        return result.graph, list(result.removed_nodes)
    return result.graph
