# tetrakis_sim/plot.py

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings
from typing import Any, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Optional plotting backends
# ---------------------------------------------------------------------------

_matplotlib_pkg = importlib.util.find_spec("matplotlib")
_mpl_spec = None
if _matplotlib_pkg is not None:
    _mpl_spec = importlib.util.find_spec("matplotlib.pyplot")

if _mpl_spec is not None:
    plt = importlib.import_module("matplotlib.pyplot")
else:  # pragma: no cover - only triggered when matplotlib is missing

    class _AxesStub:
        def add_patch(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class _MatplotlibStub:
        """Minimal stub providing the attributes used in this module."""

        def Circle(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"args": _args, "kwargs": _kwargs}

        def gca(self) -> _AxesStub:
            return _AxesStub()

    plt = _MatplotlibStub()

_HAS_MATPLOTLIB = _mpl_spec is not None
_WARNED_NO_MATPLOTLIB = False


def _warn_no_matplotlib() -> None:
    global _WARNED_NO_MATPLOTLIB
    if not _WARNED_NO_MATPLOTLIB and os.getenv("TETRAKIS_SIM_WARN_NO_PLOTTING") == "1":
        warnings.warn(
            "matplotlib is not installed; tetrakis_sim plotting functions will act as no-ops.",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED_NO_MATPLOTLIB = True

if _HAS_MATPLOTLIB:
    _mplot3d_spec = importlib.util.find_spec("mpl_toolkits.mplot3d")
    if _mplot3d_spec is not None:
        importlib.import_module("mpl_toolkits.mplot3d")


_plotly_pkg = importlib.util.find_spec("plotly")
_plotly_spec = None
if _plotly_pkg is not None:
    _plotly_spec = importlib.util.find_spec("plotly.graph_objects")

if _plotly_spec is not None:
    go = importlib.import_module("plotly.graph_objects")
else:  # pragma: no cover - only triggered when plotly is missing

    class _PlotlyTraceStub:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.args = _args
            self.kwargs = _kwargs

    class _PlotlyFigureStub:
        def add_trace(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def update_layout(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def write_html(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def write_image(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class _PlotlyStub:
        Figure = _PlotlyFigureStub
        Scatter3d = _PlotlyTraceStub
        Layout = _PlotlyTraceStub

    go = _PlotlyStub()

_HAS_PLOTLY = _plotly_spec is not None
_WARNED_NO_PLOTLY = False


def _warn_no_plotly() -> None:
    global _WARNED_NO_PLOTLY
    if not _WARNED_NO_PLOTLY and os.getenv("TETRAKIS_SIM_WARN_NO_PLOTTING") == "1":
        warnings.warn(
            "plotly is not installed; 3-D plotting helpers will act as no-ops.",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED_NO_PLOTLY = True

def plot_floor_with_circle(
    G: nx.Graph,
    z: int,
    center: Tuple[int, int, int],
    radius: float,
    highlight_nodes: Optional[List[Any]] = None,
    boundary_nodes: Optional[List[Any]] = None,
    data: Optional[dict] = None,
    figsize: Tuple[int, int] = (7, 7),
    show: bool = True,
    save: Optional[str] = None
):
    """
    Plots a single floor (z layer) of a 3D tetrakis lattice, optionally
    highlighting black hole (removed) nodes, the event horizon, and overlaying
    an analytical event horizon circle. Supports node coloring via 'data'.
    """
    if not _HAS_MATPLOTLIB:
        _warn_no_matplotlib()
        return None

    nodes_on_layer = [n for n in G if n[2] == z]
    H = G.subgraph(nodes_on_layer)
    def node_pos(node):
        r, c, z, q = node
        offset = 0.18 * "ABCD".index(q)
        return (r + offset, c + offset)
    pos = {n: node_pos(n) for n in nodes_on_layer}

    plt.figure(figsize=figsize)
    nx.draw(H, pos, node_size=60, with_labels=False, alpha=0.7)

    # Overlay color by data/amplitude if provided
    if data:
        nodes = list(H.nodes)
        xs, ys = zip(*[pos[n] for n in nodes])
        vals = [data.get(n, 0.0) for n in nodes]
        sc = plt.scatter(xs, ys, c=vals, cmap='coolwarm', s=120, zorder=20)
        plt.colorbar(sc, label="Wave amplitude")

    # Plot removed (black hole) nodes as empty circles
    if highlight_nodes:
        nodes = [n for n in highlight_nodes if len(n) > 2 and n[2] == z]
        if nodes:
            def raw_node_pos(node):
                r, c, z, q = node
                offset = 0.18 * "ABCD".index(q)
                return (r + offset, c + offset)
            xs, ys = zip(*[raw_node_pos(n) for n in nodes])
            plt.scatter(xs, ys, s=200, c='none', edgecolors='black', linewidths=2, label='Black Hole (removed)', zorder=10)
    # Event horizon nodes
    if boundary_nodes:
        nodes = [n for n in boundary_nodes if n[2] == z]
        if nodes:
            xs, ys = zip(*[pos[n] for n in nodes])
            plt.scatter(xs, ys, s=200, c='gold', edgecolors='red', label='Event Horizon', zorder=11)

    # Overlay event horizon as a circle
    r0, c0, _ = center
    circle = plt.Circle((r0, c0), radius, color='gold', fill=False, linewidth=2, linestyle='--', alpha=0.6)
    plt.gca().add_patch(circle)
    plt.title(f"Tetrakis Lattice – Floor z={z} (Analytical Horizon)")
    plt.axis('equal')
    plt.legend()
    if save is not None:
        plt.savefig(save, dpi=180, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_lattice(
    G: nx.Graph,
    data=None,
    title: Optional[str] = None,
    figsize=(8, 8),
    save: Optional[str] = None,
    layer: Optional[int] = None,
):
    """
    Plot a general 2D or 3D lattice (shows only present nodes/edges).
    Data can be a dict mapping nodes to color/size.
    For 3D lattices, provide ``layer`` to plot a single z-slice.
    """
    if not _HAS_MATPLOTLIB:
        _warn_no_matplotlib()
        return None

    plt.figure(figsize=figsize)
    pos = None
    nodes = list(G.nodes)
    if not nodes:
        return None

    if all(len(n) == 3 for n in nodes):  # 2D: (r, c, q)
        def node_pos(node):
            r, c, q = node
            offset = 0.18 * "ABCD".index(q)
            return (r + offset, c + offset)
        plot_nodes = nodes
        pos = {n: node_pos(n) for n in plot_nodes}
        H = G
    elif all(len(n) == 4 for n in nodes):  # 3D: (r, c, z, q)
        if layer is None:
            print(
                "Warning: 3D lattice detected. Provide `layer` or use "
                "plot_floor_with_circle/plot_lattice_3d."
            )
            return None
        plot_nodes = [n for n in nodes if n[2] == layer]
        if not plot_nodes:
            print(f"Warning: No nodes found on layer {layer}.")
            return None
        def node_pos(node):
            r, c, z, q = node
            offset = 0.18 * "ABCD".index(q)
            return (r + offset, c + offset)
        pos = {n: node_pos(n) for n in plot_nodes}
        H = G.subgraph(plot_nodes)
    else:
        raise ValueError("Unsupported node format for plot_lattice")

    nx.draw(H, pos, node_size=60, with_labels=False, alpha=0.7)
    if data:
        # For node coloring, overlay scatter
        xs, ys = zip(*[pos[n] for n in plot_nodes])
        colors = [data.get(n, 0.5) for n in plot_nodes]
        plt.scatter(xs, ys, c=colors, cmap='viridis', s=100)
    if title:
        plt.title(title)
    plt.axis('equal')
    if save is not None:
        plt.savefig(save, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()

# Optionally: Add stub for FFT and wave plotting


def plot_fft(freq, spectrum, node=None, values=None):
    """
    Plot the FFT amplitude spectrum. Optionally shows the time series.
    """
    if not _HAS_MATPLOTLIB:
        _warn_no_matplotlib()
        return None

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(freq, spectrum)
    plt.title(f"FFT amplitude{f' (node={node})' if node else ''}")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    if values is not None:
        plt.plot(values)
        plt.title("Node value over time")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_wave(time_steps, node_values, node=None):
    """
    Plot a time-series for a single node in a wave simulation.
    node_values should be a dict mapping node to array of values.
    """
    if not _HAS_MATPLOTLIB:
        _warn_no_matplotlib()
        return None

    if node is None:
        node = list(node_values.keys())[0]
    plt.plot(time_steps, node_values[node])
    plt.title(f"Wave at node {node}")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.show()


def plot_lattice_3d(
    G,
    horizon_nodes=None,
    removed_nodes=None,
    filename_html=None,
    filename_img=None,
    marker_size=3,
    title='Tetrakis Lattice (3D)',
):
    """
    Interactive 3D scatter plot of all nodes in a 3D tetrakis lattice.
    Optionally highlights event horizon nodes (in gold) and removed (black hole) nodes (as red or empty).
    
    Parameters:
        G: networkx.Graph
        horizon_nodes: list of nodes to highlight as event horizon (gold)
        removed_nodes: list of nodes (e.g., black hole, shown as red)
        filename_html: if set, saves the plot as interactive HTML
        filename_img: if set, saves as static PNG/SVG (requires kaleido)
        marker_size: integer, size of plotted markers
    title: plot title
    """
    if not _HAS_PLOTLY:
        _warn_no_plotly()
        return None

    # Prepare data
    x, y, z, color, symbol = [], [], [], [], []
    for node in G.nodes:
        r, c, zz, q = node
        offset = 0.18 * "ABCD".index(q)
        x.append(r + offset)
        y.append(c + offset)
        z.append(zz)
        if horizon_nodes and node in horizon_nodes:
            color.append('gold')
            symbol.append('diamond')
        else:
            color.append('blue')
            symbol.append('circle')
    # Plot removed nodes (as hollow circles or red)
    removed_x, removed_y, removed_z = [], [], []
    if removed_nodes:
        for node in removed_nodes:
            if len(node) >= 4:
                r, c, zz, q = node
                offset = 0.18 * "ABCD".index(q)
                removed_x.append(r + offset)
                removed_y.append(c + offset)
                removed_z.append(zz)
    # Build figure
    fig = go.Figure()
    # Plot lattice nodes
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=marker_size, color=color, symbol=symbol, opacity=0.8),
        name='Lattice Nodes',
        text=[str(node) for node in G.nodes]
    ))
    # Plot removed nodes (if any)
    if removed_nodes:
        fig.add_trace(go.Scatter3d(
            x=removed_x, y=removed_y, z=removed_z,
            mode='markers',
            marker=dict(size=marker_size+2, color='red', symbol='x', opacity=0.8, line=dict(width=2)),
            name='Black Hole (removed)'
        ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Row',
            yaxis_title='Col',
            zaxis_title='Layer'
        ),
        legend=dict(itemsizing='constant')
    )
    fig.show()
    # Optionally save outputs
    if filename_html:
        fig.write_html(filename_html)
    if filename_img:
        fig.write_image(filename_img)


# ---------------------------------------------------------------------------
# Generic 3-D scatter for any graph that stores Cartesian positions
# ---------------------------------------------------------------------------

def plot_3d_graph(G, node_size: int = 10, title: str = "3-D graph") -> None:
    """
    Scatter-plot any NetworkX graph whose nodes carry
    ``'pos' = (x, y, z)`` coordinates.

    Parameters
    ----------
    G : networkx.Graph
        Graph with a ``pos`` attribute on every node.
    node_size : int, optional
        Marker size, by default 10.
    title : str, optional
        Plot title.
    """
    if not _HAS_MATPLOTLIB:
        _warn_no_matplotlib()
        return None

    xs, ys, zs = zip(*(G.nodes[n]["pos"] for n in G))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=node_size)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.show()
