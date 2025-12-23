
from __future__ import annotations

import warnings
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np

from .defects import apply_defect as _core_apply_defect
from .plot import plot_fft as _plot_fft
from .plot import plot_lattice as _plot_lattice


class SimulationHistory(list):
    """List-like container returned by :func:`run_wave_sim`."""

    def __init__(
        self,
        states: Iterable[Dict[Hashable, float]],
        *,
        dt: float,
        wave_speed: float,
        metadata: Optional[Dict[str, float | bool]] = None,
    ) -> None:
        super().__init__(states)
        self.dt = dt
        self.wave_speed = wave_speed
        self.metadata = metadata or {}


def _coerce_initial_kick(nodes: List[Hashable], initial_data: Optional[Dict]) -> Dict:
    """Ensure the lattice has an initial displacement to propagate."""

    u = {n: 0.0 for n in nodes}
    if initial_data:
        u.update(initial_data)
    elif nodes:
        u[nodes[len(nodes) // 2]] = 1.0
    return u


def run_wave_sim(
    G,
    steps: int = 100,
    initial_data: Optional[Dict] = None,
    c: float = 1.0,
    dt: float = 0.2,
    damping: float = 0.0,
) -> SimulationHistory:
    """Simulate a discrete wave equation on the given lattice.

    The method performs a stability check using a simple CFL-like criterion.
    If the requested time step would be unstable it is automatically clamped
    and a :class:`RuntimeWarning` is emitted.
    """

    nodes = list(G.nodes)
    u = _coerce_initial_kick(nodes, initial_data)
    uprev = {n: 0.0 for n in nodes}

    max_degree = max((G.degree[n] for n in nodes), default=0)
    effective_dt = float(dt)
    metadata: Dict[str, float | bool] = {
        "steps": steps,
        "requested_dt": dt,
        "wave_speed": c,
        "max_degree": max_degree,
        "damping": damping,
    }

    if max_degree > 0 and c > 0.0:
        cfl_limit = float(1.0 / np.sqrt(max_degree))
        metadata["stability_limit"] = cfl_limit
        if c * effective_dt > cfl_limit:
            effective_dt = cfl_limit / c
            metadata["stability_adjusted"] = True
            metadata["effective_dt"] = float(effective_dt)
            warnings.warn(
                (
                    "Requested time-step is unstable for the current lattice "
                    f"(c*dt={c * dt:.3f} > {cfl_limit:.3f}). Clamping dt to "
                    f"{effective_dt:.3f}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
    metadata.setdefault("stability_adjusted", False)
    metadata.setdefault("effective_dt", float(effective_dt))

    history: List[Dict[Hashable, float]] = [u.copy()]
    coeff = (c * effective_dt) ** 2
    for _ in range(steps):
        unew = {}
        for n in nodes:
            neighbor_sum = sum(u[nb] for nb in G.neighbors(n))
            deg = G.degree[n]
            lap = neighbor_sum - deg * u[n]

            mass = float(G.nodes[n].get("mass", 1.0))
            potential = float(G.nodes[n].get("potential", 0.0))

            if mass <= 0.0:
                mass = 1.0

            unew[n] = (
                2 * u[n]
                - uprev[n]
                + (coeff / mass) * lap
                - coeff * potential * u[n]
                - damping * (u[n] - uprev[n])
            )

        uprev, u = u, unew
        history.append(u.copy())
            





    return SimulationHistory(history, dt=effective_dt, wave_speed=c, metadata=metadata)


def apply_defect(G, defect_type: str = "none", **kwargs) -> Tuple:
    """Convenience wrapper around :func:`tetrakis_sim.defects.apply_defect`."""

    return _core_apply_defect(G, defect_type=defect_type, return_removed=True, **kwargs)


def run_fft(
    history: Iterable[Dict[Hashable, float]],
    node: Optional[Hashable] = None,
    *,
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the FFT for a node across the simulation history."""

    history_list = list(history)
    if not history_list:
        raise ValueError("Simulation history is empty")

    sample_state = history_list[0]
    if node is None:
        node = next(iter(sample_state.keys()))

    inferred_dt = dt
    if inferred_dt is None and hasattr(history, "dt"):
        inferred_dt = getattr(history, "dt")
    if inferred_dt is None:
        inferred_dt = 1.0

    values = np.array([state.get(node, 0.0) for state in history_list])
    spectrum = np.fft.fft(values)
    freq = np.fft.fftfreq(len(values), d=inferred_dt)
    return freq, np.abs(spectrum), values


def plot_lattice(G, data=None, **kwargs):
    """Re-export for backwards compatibility with pre-0.2 physics API."""

    return _plot_lattice(G, data=data, **kwargs)


def plot_fft(freq, spectrum, *, node=None, values=None, **kwargs):
    """Re-export for backwards compatibility with pre-0.2 physics API."""

    return _plot_fft(freq, spectrum, node=node, values=values, **kwargs)
