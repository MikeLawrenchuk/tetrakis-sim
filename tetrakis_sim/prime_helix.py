# tetrakis_sim/prime_helix.py
"""
Prime-ring helix
================
Builds a helical stack of L¹-norm “diamond” rings whose radii are the prime
numbers 2, 3, 5, 7, 11, …  Each ring is rotated by a fixed increment and
lifted by a constant pitch, giving a 3-D spiral that climbs to infinity.

Public API
----------
add_prime_helix(G, n_rings=100, dtheta=math.radians(1), pitch=1.5)
"""

from __future__ import annotations
import math
from typing import List

import numpy as np
import networkx as nx

__all__ = ["add_prime_helix"]


# ---------------------------------------------------------------------------
# Prime sequence helper
# ---------------------------------------------------------------------------

def _first_primes(count: int) -> List[int]:
    """Return the first ``count`` prime numbers."""

    if count <= 0:
        return []

    primes: List[int] = []
    candidate = 2

    while len(primes) < count:
        is_prime = True
        limit = math.isqrt(candidate)
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break

        if is_prime:
            primes.append(candidate)

        candidate += 1 if candidate == 2 else 2  # skip even numbers > 2

    return primes


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def _diamond_L1(radius: float) -> np.ndarray:
    """Return 5×2 array of vertices for ‖(x,y)‖₁ = radius (closed path)."""
    return np.array([
        ( radius, 0),
        ( 0,  radius),
        (-radius, 0),
        ( 0, -radius),
        ( radius, 0),
    ], dtype=float)


def _rotated_lifted(radius: float, theta: float, z: float) -> np.ndarray:
    """
    Rotate the diamond in-plane by *theta* and append height *z*.

    Returns
    -------
    verts : ndarray, shape (5, 3)
        Ordered vertices (x,y,z) tracing the diamond.
    """
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    xy = (R @ _diamond_L1(radius).T).T        # (5,2)
    zcol = np.full((xy.shape[0], 1), z)
    return np.hstack([xy, zcol])


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def add_prime_helix(
    G: nx.Graph,
    n_rings: int = 100,
    dtheta: float = math.radians(1),
    pitch: float = 1.5,
) -> None:
    """
    Append `n_rings` rotated prime diamonds to graph *G*.

    Each node receives attributes:
    -------------------------------
    pos  : tuple[float, float, float]   3-D coordinates
    ring : int                          ring index (0 = p₂)
    prime_radius : int                  prime radius (2, 3, 5, …)
    """
    primes = _first_primes(n_rings)

    for k, p in enumerate(primes):
        verts = _rotated_lifted(p, k * dtheta, k * pitch)

        prev_id = None
        for j, (x, y, z) in enumerate(verts):
            node_id = (k, j)                   # unique & hashable
            G.add_node(node_id,
                       pos=(float(x), float(y), float(z)),
                       ring=k,
                       prime_radius=p)
            if prev_id is not None:
                G.add_edge(prev_id, node_id, ring=k)
            prev_id = node_id
