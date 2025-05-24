def apply_wedge_defect(G, center=None):
    """
    Apply a +45° wedge deficit at the grid center.
    Removes edge between center 'A' and 'B' nodes.
    """
    if center is None:
        # Infer center from any node
        rows = [r for r, _, _ in G.nodes]
        cols = [c for _, c, _ in G.nodes]
        center = (rows[len(rows)//2], cols[len(cols)//2])
    try:
        G.remove_edge((center[0], center[1], "A"), (center[0], center[1], "B"))
    except Exception:
        print(f"No edge found at {center} to remove.")
    return G

def apply_defect(G, defect_type="wedge", **kwargs):
    """
    Dispatch to the correct defect function.
    """
    if defect_type == "wedge":
        return apply_wedge_defect(G, **kwargs)
    # Add more defect types here in the future
    return G


def run_wave_sim(G):
    pass

def run_fft(G):
    pass

def plot_lattice(G, data=None):
    pass

def plot_fft(data):
    pass
