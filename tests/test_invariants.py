# tests/test_invariants.py

import math
import networkx as nx
import numpy as np
import pytest

from tetrakis_sim.lattice import build_sheet
from tetrakis_sim.defects import apply_blackhole_defect
from tetrakis_sim.physics import SimulationHistory, run_fft, run_wave_sim
from tetrakis_sim.defects import find_event_horizon



def _rotate90_2d(node, size: int):
    """Rotate (r,c,q) by 90° around the grid: (r,c) -> (c, size-1-r)."""
    r, c, q = node
    return (c, size - 1 - r, q)


def test_lattice_rotation_preserves_graph_structure_2d():
    """
    Symmetry invariant:
    A 90° rotation of coordinates should produce an isomorphic graph.
    """
    size = 4
    G = build_sheet(size=size, dim=2)

    mapping = {n: _rotate90_2d(n, size=size) for n in G.nodes}
    G_rot = nx.relabel_nodes(G, mapping, copy=True)

    # Strong check: exact graph isomorphism
    assert nx.is_isomorphic(G, G_rot)

    # Extra cheap invariant: degree multiset unchanged
    assert sorted(dict(G.degree()).values()) == sorted(dict(G_rot.degree()).values())


def test_blackhole_defect_removes_exact_expected_nodes_2d():
    """
    Defect invariant:
    For a 2D blackhole defect, the removed node set should match the
    mathematical predicate: hypot(r-r0, c-c0) < radius (for all q in ABCD).
    """
    size = 5
    center = (2, 2)
    radius = 1.5

    G = build_sheet(size=size, dim=2)
    removed = apply_blackhole_defect(G, center=center, radius=radius)

    # Compute the expected removed set directly from the definition in defects.py
    r0, c0 = center
    expected = {
        (r, c, q)
        for r in range(size)
        for c in range(size)
        for q in "ABCD"
        if math.hypot(r - r0, c - c0) < radius
    }

    assert set(removed) == expected
    # And as a sanity check: none of those nodes remain in the graph
    assert not (set(removed) & set(G.nodes))


def test_wave_sim_enforces_cfl_limit_and_stays_bounded():
    """
    Stability invariant:
    If dt is too large, run_wave_sim should clamp it to stability_limit/c,
    record that in metadata, and produce finite, non-exploding values.
    """
    G = build_sheet(size=3, dim=2)

    # Choose a deliberately-too-large dt so clamping must occur
    with pytest.warns(RuntimeWarning, match=r"Clamping dt"):
        history = run_wave_sim(G, steps=25, c=1.0, dt=10.0, damping=0.0)

    assert history.metadata["stability_adjusted"] is True

    # From physics.py: stability_limit is 1/sqrt(max_degree) and effective_dt=stability_limit/c
    max_degree = history.metadata["max_degree"]
    expected_limit = 1.0 / math.sqrt(max_degree) if max_degree > 0 else 0.0
    assert history.metadata["stability_limit"] == pytest.approx(expected_limit)
    assert history.dt == pytest.approx(expected_limit / 1.0)  # c=1.0

    # Boundedness: not NaN/inf, and amplitudes do not blow up absurdly
    all_vals = []
    for state in history:
        all_vals.extend(state.values())

    arr = np.asarray(all_vals, dtype=float)
    assert np.isfinite(arr).all()
    assert np.max(np.abs(arr)) < 1e3  # generous cap; catches true explosions



def test_fft_detects_known_sine_frequency():
    """
    FFT sanity invariant:
    Feed a pure sine wave at frequency f0 into a SimulationHistory with dt.
    The dominant FFT bin should land near +/- f0 (within frequency resolution).
    """
    f0 = 2.0      # Hz
    dt = 0.1      # seconds
    N = 100       # samples -> resolution = 1/(N*dt) = 0.1 Hz
    t = np.arange(N) * dt

    values = np.sin(2.0 * np.pi * f0 * t)
    states = [{0: float(v)} for v in values]
    history = SimulationHistory(states, dt=dt, wave_speed=1.0)

    freq, spectrum, _ = run_fft(history, node=0)
    dom = float(freq[int(np.argmax(spectrum))])

    resolution = 1.0 / (N * dt)
    assert abs(abs(dom) - f0) <= resolution + 1e-9


def test_event_horizon_nodes_touch_removed_nodes_2d():
    size = 7
    center = (3, 3)
    radius = 1.9

    G0 = build_sheet(size=size, dim=2)  # pre-defect graph for adjacency
    G1 = build_sheet(size=size, dim=2)  # graph that will actually be mutated

    removed = set(apply_blackhole_defect(G1, center=center, radius=radius))

    horizon = set(
        find_event_horizon(G1, removed, radius=radius, center=center, adjacency_graph=G0)
    )
    assert horizon, "Expected a non-empty horizon for this radius/center."

    for n in horizon:
        assert any(nb in removed for nb in G0.neighbors(n)), (
            f"Horizon node {n} has no removed neighbors in the pre-defect graph."
        )


def test_event_horizon_excludes_removed_nodes():
    size = 7
    center = (3, 3)
    radius = 1.9

    G0 = build_sheet(size=size, dim=2)
    G1 = build_sheet(size=size, dim=2)
    removed = set(apply_blackhole_defect(G1, center=center, radius=radius))

    horizon = set(
        find_event_horizon(G1, removed, radius=radius, center=center, adjacency_graph=G0)
    )
    assert horizon.isdisjoint(removed), "Horizon must not include removed nodes."


def test_blackhole_defect_removes_all_layers_for_xy_in_3d():
    size = 6
    center = (2.0, 2.0)
    radius = 1.6
    layers = 4

    G = build_sheet(size=size, dim=3, layers=layers)
    removed = set(apply_blackhole_defect(G, center=center, radius=radius))

    # Pick one (x,y) that should definitely be inside the radius
    x, y = 2, 2
    for z in range(layers):
        for q in "ABCD":
            assert (x, y, z, q) in removed


def test_fft_frequency_invariant_under_sampling_rate_change():
    f0 = 3.0  # Hz

    # Case A
    dt_a = 0.05
    N_a = 200
    t_a = np.arange(N_a) * dt_a
    sig_a = np.sin(2.0 * np.pi * f0 * t_a)
    hist_a = SimulationHistory([{0: float(v)} for v in sig_a], dt=dt_a, wave_speed=1.0)

    # Case B (different dt, similar total duration)
    dt_b = 0.1
    N_b = 100
    t_b = np.arange(N_b) * dt_b
    sig_b = np.sin(2.0 * np.pi * f0 * t_b)
    hist_b = SimulationHistory([{0: float(v)} for v in sig_b], dt=dt_b, wave_speed=1.0)

    freq_a, spec_a, _ = run_fft(hist_a, node=0)
    freq_b, spec_b, _ = run_fft(hist_b, node=0)

    dom_a = float(freq_a[int(np.argmax(spec_a))])
    dom_b = float(freq_b[int(np.argmax(spec_b))])

    # Allow tolerance at each frequency resolution
    res_a = 1.0 / (N_a * dt_a)
    res_b = 1.0 / (N_b * dt_b)

    assert abs(abs(dom_a) - f0) <= res_a + 1e-9
    assert abs(abs(dom_b) - f0) <= res_b + 1e-9

