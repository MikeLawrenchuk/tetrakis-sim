
import numpy as np

def run_wave_sim(
    G,
    steps=100,
    initial_data=None,
    c=1.0,          # wave speed
    dt=0.2,         # time step size
    damping=0.0,    # optional: add a little to see dissipation
):
    """
    Discrete wave simulation on a graph (2D or 3D).
    Uses a simple explicit finite-difference (FDTD) scheme.
    
    Args:
        G: networkx.Graph
        steps: number of time steps to run
        initial_data: dict {node: value} (displacement at t=0)
        c: wave speed (affects stability)
        dt: time step size
        damping: 0 = no damping, >0 = dissipates wave energy
        
    Returns:
        hist: list of dicts (node -> displacement at each step)
    """
    nodes = list(G.nodes)
    # 1. Initialize states (u: current, uprev: previous)
    u = {n: 0.0 for n in nodes}
    uprev = {n: 0.0 for n in nodes}
    if initial_data:
        for n, val in initial_data.items():
            u[n] = val
    # Small kick for a central node if not provided
    if not initial_data:
        n0 = nodes[len(nodes)//2]
        u[n0] = 1.0

    hist = [u.copy()]
    for step in range(steps):
        unew = {}
        for n in nodes:
            neighbor_sum = sum(u[nb] for nb in G.neighbors(n))
            deg = G.degree[n]
            # Discrete wave equation on a graph
            # u_next = 2u - u_prev + (c*dt)^2 * (Laplace_u)
            lap = neighbor_sum - deg*u[n]
            unew[n] = (
                2*u[n] - uprev[n]
                + (c*dt)**2 * lap
                - damping * (u[n] - uprev[n])
            )
        uprev, u = u, unew
        hist.append(u.copy())
    return hist



def apply_defect(G, defect_type="none"):
    pass





def run_fft(history, node=None):
    """
    Compute FFT for a specific node’s value over time.
    history: list of dicts (node -> value), output from run_wave_sim
    node: the node to analyze (tuple). If None, picks first node.
    Returns: frequencies, spectrum (abs value), and the time-series values
    """
    if node is None:
        node = list(history[0].keys())[0]
    values = np.array([state.get(node, 0.0) for state in history])
    spectrum = np.fft.fft(values)
    freq = np.fft.fftfreq(len(values))
    return freq, np.abs(spectrum), values


def plot_lattice(G, data=None):
    pass

def plot_fft(data):
    pass
