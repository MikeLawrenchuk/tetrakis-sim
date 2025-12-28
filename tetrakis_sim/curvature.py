"""tetrakis_sim.curvature

Discrete curvature utilities for embedded (geometric) graphs.

This module is intentionally lightweight: it provides a practical,
publishable-by-default curvature proxy (angle deficit) for triangulated
graphs with 2D (or 3D) node coordinates.

Core idea (Regge/triangle mesh intuition):
    K(v) ≈ 2π - Σ_{triangles incident to v} angle_v(triangle)

For graphs with boundaries (including hole boundaries), the boundary
version is:
    K_boundary(v) ≈ π - Σ angles

We auto-detect boundary vertices via "boundary edges":
an edge (u,v) is boundary if it participates in exactly one triangle.

Notes:
- This is an *analogue metric* for your lattice geometry; it is not GR.
- It is only meaningful when node coordinates represent an embedding.
- Tetrakis lattices are naturally triangulated, so this works well.

Dependencies: numpy, networkx (both already used by Tetrakis-Sim).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
)

import networkx as nx
import numpy as np

Pos = Sequence[float] | np.ndarray
PosMap = Mapping[Any, Pos]


def _node_key(n: Any) -> str:
    # Avoid TypeError on mixed node types in sorting/dedup.
    return repr(n)


def get_pos(
    G: nx.Graph,
    n: Any,
    *,
    pos: PosMap | None = None,
    pos_attr_candidates: Sequence[str] = ("pos", "xyz", "coords", "coord", "position"),
) -> np.ndarray:
    """Return node position as a numpy array of shape (d,).

    The function tries, in order:
      1) a supplied pos mapping,
      2) node attributes in pos_attr_candidates,
      3) (x,y) or (x,y,z) attributes.

    Raises:
        KeyError if no position is found.
    """
    if pos is not None:
        p = pos[n]
        return np.asarray(p, dtype=float)

    data = G.nodes[n]

    for attr in pos_attr_candidates:
        if attr in data:
            p = data[attr]
            if isinstance(p, list | tuple | np.ndarray) and len(p) >= 2:
                return np.asarray(p, dtype=float)

    if "x" in data and "y" in data:
        if "z" in data:
            return np.asarray([data["x"], data["y"], data["z"]], dtype=float)
        return np.asarray([data["x"], data["y"]], dtype=float)

    raise KeyError(
        f"No position found for node {n!r}. Provide pos=... or store node attrs like 'pos' or 'x','y'."
    )


def triangles(G: nx.Graph) -> list[tuple[Any, Any, Any]]:
    """Return all triangles (3-cycles) as sorted node triples.

    This is a purely graph-theoretic triangle enumeration. For large graphs,
    this can be expensive; for Tetrakis sizes used in demos/sweeps it is fine.
    """
    tris: set[tuple[Any, Any, Any]] = set()

    for u in G.nodes:
        nbrs = list(G.neighbors(u))
        # Check all neighbor pairs (v,w). If (v,w) is an edge, (u,v,w) is a triangle.
        for i in range(len(nbrs)):
            v = nbrs[i]
            for j in range(i + 1, len(nbrs)):
                w = nbrs[j]
                if G.has_edge(v, w):
                    tri = tuple(sorted((u, v, w), key=_node_key))
                    tris.add(tri)

    return sorted(
        tris, key=lambda t: (_node_key(t[0]), _node_key(t[1]), _node_key(t[2]))
    )


def _angle(pu: np.ndarray, pv: np.ndarray, pw: np.ndarray) -> float:
    """Angle at pu in triangle (pu,pv,pw) in radians."""
    a = pv - pu
    b = pw - pu
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    # Clamp for numerical stability
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c))


def edge_triangle_count(G: nx.Graph, u: Any, v: Any) -> int:
    """Number of triangles that contain edge (u,v)."""
    # Common neighbors form triangles (u,v,w).
    # In a well-behaved planar triangulation, interior edges have count=2, boundary edges count=1.
    if not G.has_edge(u, v):
        return 0
    Nu = set(G.neighbors(u))
    Nv = set(G.neighbors(v))
    return len(Nu & Nv)


def boundary_edges(G: nx.Graph) -> set[tuple[Any, Any]]:
    """Return edges that appear to be boundary edges (triangle count == 1)."""
    b: set[tuple[Any, Any]] = set()
    for u, v in G.edges:
        c = edge_triangle_count(G, u, v)
        if c == 1:
            # Normalize edge representation
            e = tuple(sorted((u, v), key=_node_key))
            b.add(e)
    return b


def boundary_vertices(G: nx.Graph) -> set[Any]:
    """Vertices incident to a boundary edge."""
    b_edges = boundary_edges(G)
    verts: set[Any] = set()
    for u, v in b_edges:
        verts.add(u)
        verts.add(v)
    return verts


def angle_sum_at_node(
    G: nx.Graph,
    u: Any,
    *,
    pos: PosMap | None = None,
) -> float:
    """Sum of triangle angles incident at node u.

    Computes:
        Σ_{(u,v,w) triangle} angle_u(u,v,w)
    """
    pu = get_pos(G, u, pos=pos)
    nbrs = list(G.neighbors(u))

    total = 0.0
    for i in range(len(nbrs)):
        v = nbrs[i]
        for j in range(i + 1, len(nbrs)):
            w = nbrs[j]
            if G.has_edge(v, w):
                pv = get_pos(G, v, pos=pos)
                pw = get_pos(G, w, pos=pos)
                total += _angle(pu, pv, pw)

    return float(total)


def angle_deficit(
    G: nx.Graph,
    nodes: Iterable[Any] | None = None,
    *,
    pos: PosMap | None = None,
    use_boundary_correction: bool = True,
) -> dict[Any, float]:
    """Compute angle deficit curvature estimate for selected nodes.

    Args:
        G: embedded graph (nodes must have positions).
        nodes: nodes to evaluate (default: all nodes).
        pos: optional mapping node -> position.
        use_boundary_correction: if True, uses π - sum_angles for boundary vertices
            and 2π - sum_angles for interior vertices.

    Returns:
        dict: node -> deficit (radians). Positive deficit means "positive" curvature.

    Interpretation:
        - In a flat triangulated interior: deficit ~ 0.
        - Near a wedge / missing region boundary: deficit typically > 0.
    """
    if nodes is None:
        nodes = list(G.nodes)

    bverts: set[Any] = boundary_vertices(G) if use_boundary_correction else set()

    out: dict[Any, float] = {}
    for u in nodes:
        s = angle_sum_at_node(G, u, pos=pos)
        if use_boundary_correction and u in bverts:
            out[u] = float(math.pi - s)
        else:
            out[u] = float(2.0 * math.pi - s)

    return out


@dataclass(frozen=True)
class CurvatureSummary:
    """Compact summary statistics for a set of per-node deficits."""

    n: int
    mean: float
    std: float
    min: float
    max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": float(self.n),
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min),
            "max": float(self.max),
        }


def summarize_deficits(deficits: Mapping[Any, float]) -> CurvatureSummary:
    vals = np.asarray(list(deficits.values()), dtype=float)
    if vals.size == 0:
        return CurvatureSummary(
            n=0, mean=float("nan"), std=float("nan"), min=float("nan"), max=float("nan")
        )
    return CurvatureSummary(
        n=int(vals.size),
        mean=float(vals.mean()),
        std=float(vals.std(ddof=0)),
        min=float(vals.min()),
        max=float(vals.max()),
    )


def ego_nodes(G: nx.Graph, centers: Iterable[Any], hops: int = 2) -> set[Any]:
    """Return nodes within `hops` graph distance from any of `centers`."""
    out: set[Any] = set()
    for c in centers:
        if c not in G:
            continue
        d = nx.single_source_shortest_path_length(G, c, cutoff=hops)
        out.update(d.keys())
    return out


def infer_singularity_centers(G: nx.Graph) -> list[Any]:
    """Heuristic: return nodes flagged as singular or carrying a 'mass'/'potential' attribute."""
    centers: list[Any] = []
    for n, data in G.nodes(data=True):
        if data.get("singular", False):
            centers.append(n)
            continue
        if "mass" in data or "potential" in data:
            centers.append(n)
    return centers


def curvature_report(
    G: nx.Graph,
    *,
    centers: Iterable[Any] | None = None,
    hops: int = 2,
    pos: PosMap | None = None,
    use_boundary_correction: bool = True,
) -> dict[str, Any]:
    """Compute curvature summaries globally and near a defect center.

    Returns a JSON-friendly dict, intended to be dropped into your
    run metadata.
    """
    if centers is None:
        centers = infer_singularity_centers(G)

    global_def = angle_deficit(
        G, pos=pos, use_boundary_correction=use_boundary_correction
    )
    global_summary = summarize_deficits(global_def).as_dict()

    near_summary: dict[str, float] | None = None
    if centers:
        near_nodes = ego_nodes(G, centers, hops=hops)
        near_def = {n: global_def[n] for n in near_nodes if n in global_def}
        near_summary = summarize_deficits(near_def).as_dict()

    return {
        "curvature": {
            "method": "angle_deficit",
            "use_boundary_correction": bool(use_boundary_correction),
            "global": global_summary,
            "near_centers": near_summary,
            "centers": [repr(c) for c in centers] if centers else [],
            "hops": int(hops),
        }
    }
