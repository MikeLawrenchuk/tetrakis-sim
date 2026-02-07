from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tetrakis_sim.defects import apply_defect
from tetrakis_sim.evals.dataset import generate_defect_classification_jsonl
from tetrakis_sim.lattice import build_sheet
from tetrakis_sim.physics import run_wave_sim


def _center_2d(size: int) -> tuple[int, int]:
    return (size // 2, size // 2)


def _max_degree(G) -> int:
    return max((deg for _, deg in G.degree()), default=0)


def test_blackhole_removes_nodes_and_reports_removed_nodes() -> None:
    size = 11
    center = _center_2d(size)

    base = build_sheet(size=size, dim=2, layers=1)
    n0 = base.number_of_nodes()

    G, removed = apply_defect(
        base.copy(),
        defect_type="blackhole",
        return_removed=True,
        center=center,
        radius=2.0,
    )

    assert G.number_of_nodes() < n0

    base_nodes = set(base.nodes)
    removed_nodes = [x for x in removed if x in base_nodes]
    assert len(removed_nodes) > 0

    for n in removed_nodes:
        assert n not in G

    # If the defect reports removed nodes, the count should match the delta.
    assert G.number_of_nodes() == n0 - len(removed_nodes)


def test_wedge_removes_edges_not_nodes() -> None:
    size = 11
    center = _center_2d(size)

    base = build_sheet(size=size, dim=2, layers=1)
    n0 = base.number_of_nodes()
    m0 = base.number_of_edges()

    G, _removed = apply_defect(
        base.copy(),
        defect_type="wedge",
        return_removed=True,
        center=center,
    )

    assert G.number_of_nodes() == n0
    assert G.number_of_edges() < m0


def test_singularity_tags_nodes_without_removing_nodes() -> None:
    size = 11
    center = _center_2d(size)

    base = build_sheet(size=size, dim=2, layers=1)
    n0 = base.number_of_nodes()

    G, removed = apply_defect(
        base.copy(),
        defect_type="singularity",
        return_removed=True,
        center=center,
        radius=1.0,
        mass=50.0,
        potential=25.0,
        prune_edges=False,
    )

    assert G.number_of_nodes() == n0

    base_nodes = set(base.nodes)
    removed_nodes = [x for x in removed if x in base_nodes]
    assert len(removed_nodes) == 0

    singular_nodes = [n for n in G.nodes if bool(G.nodes[n].get("singular", False))]
    assert len(singular_nodes) > 0

    # At least one singular node should have non-default mass/potential.
    assert any(float(G.nodes[n].get("mass", 1.0)) != 1.0 for n in singular_nodes) or any(
        float(G.nodes[n].get("potential", 0.0)) != 0.0 for n in singular_nodes
    )


def test_run_wave_sim_clamps_dt_when_unstable() -> None:
    size = 7
    G = build_sheet(size=size, dim=2, layers=1)

    max_deg = _max_degree(G)
    assert max_deg > 0

    c = 1.0
    dt_requested = 10.0  # intentionally too large
    expected_limit = 1.0 / math.sqrt(max_deg)
    expected_dt = expected_limit / c

    with pytest.warns(RuntimeWarning):
        hist = run_wave_sim(G, steps=1, initial_data=None, c=c, dt=dt_requested, damping=0.0)

    assert hist.metadata.get("stability_adjusted") is True
    assert hist.metadata.get("stability_limit") == pytest.approx(expected_limit)
    assert hist.metadata.get("effective_dt") == pytest.approx(expected_dt)
    assert hist.dt == pytest.approx(expected_dt)


def test_run_wave_sim_does_not_clamp_dt_when_stable() -> None:
    size = 7
    G = build_sheet(size=size, dim=2, layers=1)

    max_deg = _max_degree(G)
    assert max_deg > 0

    c = 1.0
    dt_requested = 0.01
    assert c * dt_requested < (1.0 / math.sqrt(max_deg))

    hist = run_wave_sim(G, steps=1, initial_data=None, c=c, dt=dt_requested, damping=0.0)

    assert hist.metadata.get("stability_adjusted") is False
    assert hist.metadata.get("effective_dt") == pytest.approx(dt_requested)
    assert hist.dt == pytest.approx(dt_requested)


def test_eval_dataset_records_contain_core_feature_keys(tmp_path: Path) -> None:
    out = tmp_path / "defect_classification.jsonl"
    generate_defect_classification_jsonl(
        out,
        n_per_class=1,
        seed=0,
        size=7,
        dim=2,
        layers=1,
        steps=5,
        c=1.0,
        dt=0.2,
        damping=0.0,
    )

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    rec = json.loads(lines[0])
    feats = rec["features"]
    params = rec["params"]

    # Core feature contracts
    for k in [
        "n_nodes",
        "n_edges",
        "max_degree",
        "dominant_freq",
        "spectral_centroid",
        "ts_rms",
        "kick_degree",
    ]:
        assert k in feats

    # Basic dt contract: dt_used is reported and should not exceed requested dt.
    assert "dt_requested" in params
    assert "dt_used" in params
    assert float(params["dt_used"]) <= float(params["dt_requested"]) + 1e-12
