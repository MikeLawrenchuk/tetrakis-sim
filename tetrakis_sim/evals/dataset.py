from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from tetrakis_sim.defects import apply_defect
from tetrakis_sim.lattice import build_sheet
from tetrakis_sim.physics import run_fft, run_wave_sim

from .features import spectral_features, timeseries_features


def _center_for(size: int, dim: int, layers: int) -> tuple[int, ...]:
    if dim == 2:
        return (size // 2, size // 2)
    return (size // 2, size // 2, layers // 2)


def _pick_kick_node(G: Any, center: tuple[int, ...], dim: int) -> Any:
    if dim == 3:
        z = center[2]
        nodes = [n for n in G.nodes if len(n) > 3 and n[2] == z]
    else:
        nodes = list(G.nodes)

    if not nodes:
        raise RuntimeError("No nodes available to kick")

    def dist2(node: Any) -> int:
        r, c = node[:2]
        return (r - center[0]) ** 2 + (c - center[1]) ** 2

    candidates = [
        n for n in nodes if G.degree[n] > 0 and not bool(G.nodes[n].get("singular", False))
    ]
    pool = candidates if candidates else nodes
    return min(pool, key=dist2)


def _graph_features(G: Any, removed_nodes: list[Any]) -> dict[str, float]:
    n = float(G.number_of_nodes())
    m = float(G.number_of_edges())
    avg_degree = float((2.0 * m / n) if n > 0 else 0.0)
    max_degree = float(max((G.degree[v] for v in G.nodes), default=0))
    singular_count = float(sum(1 for v in G.nodes if bool(G.nodes[v].get("singular", False))))
    return {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "removed_node_count": float(len(removed_nodes)),
        "singular_node_count": singular_count,
    }


def generate_defect_classification_jsonl(
    out_path: str | Path,
    *,
    n_per_class: int = 10,
    seed: int = 0,
    size: int = 11,
    dim: int = 2,
    layers: int = 5,
    steps: int = 40,
    c: float = 1.0,
    dt: float = 0.2,
    damping: float = 0.0,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    defect_types = ["none", "wedge", "blackhole", "singularity"]

    records: list[dict[str, Any]] = []
    idx = 0

    for defect_type in defect_types:
        for _ in range(int(n_per_class)):
            idx += 1

            G = build_sheet(size=size, dim=dim, layers=layers)
            center = _center_for(size=size, dim=dim, layers=layers)

            params: dict[str, Any] = {
                "size": size,
                "dim": dim,
                "layers": layers,
                "steps": steps,
                "c": c,
                "dt": dt,
                "damping": damping,
            }

            defect_kwargs: dict[str, Any] = {}
            if defect_type == "blackhole":
                r = float(rng.choice([1.5, 2.0, 2.5]))
                defect_kwargs = {"center": center, "radius": r}
                params.update({"radius": r})
            elif defect_type == "wedge":
                if dim == 2:
                    defect_kwargs = {"center": center}
                else:
                    defect_kwargs = {"center": center[:2], "layer": center[2]}
                    params.update({"layer": int(center[2])})
            elif defect_type == "singularity":
                sr = float(rng.choice([0.0, 0.5, 1.0]))
                mass = float(rng.choice([50.0, 200.0, 1000.0]))
                pot = float(rng.choice([0.0, 25.0]))
                prune = bool(rng.choice([False, True]))
                defect_kwargs = {
                    "center": center,
                    "radius": sr,
                    "mass": mass,
                    "potential": pot,
                    "prune_edges": prune,
                }
                params.update({"radius": sr, "mass": mass, "potential": pot, "prune_edges": prune})

            removed_nodes: list[Any] = []
            if defect_type != "none":
                G, removed_nodes = apply_defect(
                    G, defect_type=defect_type, return_removed=True, **defect_kwargs
                )

            kick = _pick_kick_node(G, center=center, dim=dim)

            history = run_wave_sim(
                G,
                steps=steps,
                initial_data={kick: 1.0},
                c=c,
                dt=dt,
                damping=damping,
            )

            freq, amp, values = run_fft(history, node=kick)

            feats: dict[str, float] = {}
            feats.update(_graph_features(G, removed_nodes))
            feats.update(spectral_features(freq, amp))
            feats.update(timeseries_features(values))
            feats["kick_degree"] = float(G.degree[kick])
            feats["kick_dist2_center"] = float(
                (kick[0] - center[0]) ** 2 + (kick[1] - center[1]) ** 2
            )

            rec = {
                "id": f"dc_{idx:05d}",
                "task": "defect_classification",
                "label": defect_type,
                "params": params,
                "features": feats,
            }
            records.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    return out_path
