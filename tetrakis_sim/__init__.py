"""
tetrakis_sim
============

Core simulation and visualisation utilities for the Tetrakis-Sim project.
"""

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
from .plot import plot_3d_graph  # lightweight 3-D scatter
from .prime_helix import add_prime_helix  # geometry builder

__all__: list[str] = [
    "add_prime_helix",
    "plot_3d_graph",
]
