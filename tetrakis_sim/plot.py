# tetrakis_sim/plot.py

import matplotlib.pyplot as plt
import networkx as nx
from typing import Any, List, Optional, Tuple

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


def plot_lattice(G: nx.Graph, data=None, title: Optional[str] = None, figsize=(8,8), save: Optional[str]=None):
    """
    Plot a general 2D or 3D lattice (shows only present nodes/edges).
    Data can be a dict mapping nodes to color/size.
    """
    plt.figure(figsize=figsize)
    pos = None
    if all(len(n) == 3 for n in G.nodes):  # 2D: (r, c, q)
        def node_pos(node):
            r, c, q = node
            offset = 0.18 * "ABCD".index(q)
            return (r + offset, c + offset)
        pos = {n: node_pos(n) for n in G.nodes}
    elif all(len(n) == 4 for n in G.nodes):  # 3D: (r, c, z, q) - only show one z?
        print("Warning: 3D lattice detected. Use plot_floor_with_circle for specific floors.")
        return
    nx.draw(G, pos, node_size=60, with_labels=False, alpha=0.7)
    if data:
        # For node coloring, overlay scatter
        nodes = list(G.nodes)
        xs, ys = zip(*[pos[n] for n in nodes])
        colors = [data.get(n, 0.5) for n in nodes]
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
    if node is None:
        node = list(node_values.keys())[0]
    plt.plot(time_steps, node_values[node])
    plt.title(f"Wave at node {node}")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.show()


import plotly.graph_objects as go

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
