import math
from collections.abc import Sequence

import pytest

from tetrakis_sim.defects import apply_defect, apply_singularity_defect
from tetrakis_sim.lattice import build_sheet


def _dist(node: tuple, center: tuple[int, ...]) -> float:
    """
    Mirror the distance logic in apply_singularity_defect:
      - center=(r,c) uses node[:2]
      - center=(r,c,z) uses node[:3]
    """
    if len(center) == 2:
        r, c = node[:2]
        return math.hypot(r - center[0], c - center[1])
    r, c, z = node[:3]
    return math.sqrt((r - center[0]) ** 2 + (c - center[1]) ** 2 + (z - center[2]) ** 2)


def _tagged_nodes(G, center: tuple[int, ...], radius: float) -> Sequence[tuple]:
    return [n for n in G.nodes if _dist(n, center) <= radius]


@pytest.mark.parametrize(
    "dim,layers,center,radius",
    [
        (2, None, (3, 3), 1.01),
        (3, 4, (3, 3, 2), 1.01),
    ],
)
def test_singularity_tags_expected_nodes_and_keeps_nodes(dim, layers, center, radius):
    kwargs = {"size": 7, "dim": dim}
    if dim == 3:
        kwargs["layers"] = layers

    G = build_sheet(**kwargs)
    n0 = G.number_of_nodes()
    e0 = G.number_of_edges()

    mass = 123.0
    potential = 4.5

    result = apply_singularity_defect(
        G,
        center=center,
        radius=radius,
        mass=mass,
        potential=potential,
        prune_edges=False,
        return_result=True,
    )

    # No node deletion, and no edge deletion when prune_edges=False
    assert G.number_of_nodes() == n0
    assert G.number_of_edges() == e0

    tagged = list(_tagged_nodes(G, center, radius))
    assert tagged, "Expected at least one tagged node for this (center, radius)."

    # Implementation metadata should agree with the tagging rule
    assert result.metadata["singularity_tagged_count"] == len(tagged)

    # Tagged nodes must have the new attributes
    for n in tagged:
        attrs = G.nodes[n]
        assert attrs.get("singular") is True
        assert attrs.get("mass") == float(mass)
        assert attrs.get("potential") == float(potential)

    # Untagged nodes should not be marked singular (fresh graph assumption)
    for n in set(G.nodes) - set(tagged):
        assert G.nodes[n].get("singular", False) is False

    # Singularity defect returns no removed nodes (by design)
    assert list(result.removed_nodes) == []
    assert list(result.removed_edges) == []


def test_singularity_prune_edges_isolates_tagged_nodes_2d():
    size = 7
    center = (size // 2, size // 2)

    # radius=0 should tag nodes exactly at (center_r, center_c)
    radius = 0.0

    G = build_sheet(size=size, dim=2)
    e0 = G.number_of_edges()

    result = apply_singularity_defect(
        G,
        center=center,
        radius=radius,
        prune_edges=True,
        return_result=True,
    )

    tagged = list(_tagged_nodes(G, center, radius))
    assert (
        tagged
    ), "Expected at least one tagged node at radius=0 for an on-lattice center."

    # With prune_edges=True, all tagged nodes should become isolated (degree 0)
    for n in tagged:
        assert G.nodes[n].get("singular") is True
        assert G.degree[n] == 0

    e1 = G.number_of_edges()
    assert e1 < e0, "Edge count should decrease when pruning is enabled."
    assert (e0 - e1) == len(
        result.removed_edges
    ), "Removed-edge list should match the edge delta."

    # Every removed edge must touch a tagged node
    tagged_set = set(tagged)
    for u, v in result.removed_edges:
        assert (u in tagged_set) or (v in tagged_set)


def test_apply_defect_singularity_reports_no_removed_nodes():
    G = build_sheet(size=5, dim=2)

    G2, removed = apply_defect(
        G,
        defect_type="singularity",
        return_removed=True,
        center=(2, 2),
        radius=0.0,
        mass=1000.0,
        potential=0.0,
        prune_edges=False,
    )

    # apply_defect mutates in-place and returns the same graph object
    assert G2 is G
    assert removed == []
