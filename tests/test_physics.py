import math
import networkx as nx
import numpy as np
import pytest

from tetrakis_sim.defects import apply_wedge_defect
from tetrakis_sim.lattice import build_sheet
from tetrakis_sim.physics import (
    SimulationHistory,
    apply_defect as physics_apply_defect,
    run_fft,
    run_wave_sim,
)


def test_run_wave_sim_attaches_metadata():
    G = nx.path_graph(3)
    with pytest.warns(RuntimeWarning):
        history = run_wave_sim(G, steps=3, c=1.0, dt=5.0)

    assert isinstance(history, SimulationHistory)
    assert history.dt == pytest.approx(1 / math.sqrt(2))
    assert history.metadata["stability_adjusted"] is True
    assert history.metadata["max_degree"] == 2


def test_run_fft_uses_history_dt():
    states = [{0: 0.0}, {0: 1.0}, {0: 0.0}, {0: -1.0}]
    history = SimulationHistory(states, dt=0.5, wave_speed=1.0)

    freq, spectrum, values = run_fft(history, node=0)

    assert np.allclose(values, [0.0, 1.0, 0.0, -1.0])
    assert pytest.approx(freq[1]) == 0.5  # second bin corresponds to 1 / (N*dt)
    assert np.argmax(spectrum) in (1, 3)


def test_apply_wedge_defect_supports_3d():
    G = build_sheet(size=3, dim=3, layers=2)
    center = (1, 1, 0)
    apply_wedge_defect(G, center=center[:2], layer=center[2])

    assert not G.has_edge((1, 1, 0, "A"), (1, 1, 0, "B"))
    assert G.has_edge((1, 1, 1, "A"), (1, 1, 1, "B"))


def test_physics_apply_defect_returns_removed_nodes():
    G = build_sheet(size=5, dim=3, layers=3)
    center = (2, 2, 1)
    _, removed = physics_apply_defect(
        G, defect_type="blackhole", center=center, radius=1.5
    )

    assert removed
    for node in removed:
        assert math.dist(node[:3], center) < 1.5
