from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import networkx as nx

T = TypeVar("T")

Kind = Literal["lattice", "defect", "eval_task"]


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    kind: Kind
    fn: Callable[..., Any]
    description: str = ""


_LATTICES: dict[str, RegistryEntry] = {}
_DEFECTS: dict[str, RegistryEntry] = {}
_EVAL_TASKS: dict[str, RegistryEntry] = {}


def _bucket(kind: Kind) -> dict[str, RegistryEntry]:
    if kind == "lattice":
        return _LATTICES
    if kind == "defect":
        return _DEFECTS
    if kind == "eval_task":
        return _EVAL_TASKS
    raise ValueError(f"Unknown kind: {kind}")


def register(
    kind: Kind,
    name: str,
    fn: Callable[..., Any],
    *,
    description: str = "",
    overwrite: bool = False,
) -> None:
    """Register a callable under a kind/name.

    Parameters
    ----------
    kind:
        "lattice" | "defect" | "eval_task"
    name:
        Stable key used by CLIs/configs.
    fn:
        Callable implementing the behavior.
    description:
        Optional human-readable text for docs/help.
    overwrite:
        If False (default), re-registering an existing name raises.
    """
    if not name or not isinstance(name, str):
        raise ValueError("name must be a non-empty string")

    b = _bucket(kind)
    if not overwrite and name in b:
        raise KeyError(f"{kind} already registered: {name}")

    b[name] = RegistryEntry(name=name, kind=kind, fn=fn, description=description)


def get(kind: Kind, name: str) -> Callable[..., Any]:
    """Retrieve a previously registered callable."""
    b = _bucket(kind)
    if name not in b:
        raise KeyError(f"{kind} not registered: {name}")
    return b[name].fn


def describe(kind: Kind, name: str) -> RegistryEntry:
    """Return the metadata for a registered entry."""
    b = _bucket(kind)
    if name not in b:
        raise KeyError(f"{kind} not registered: {name}")
    return b[name]


def list_names(kind: Kind) -> list[str]:
    """List registered names for a kind."""
    return sorted(_bucket(kind).keys())


# Convenience helpers (typed wrappers) ---------------------------------


def register_lattice(
    name: str,
    fn: Callable[..., nx.Graph],
    *,
    description: str = "",
    overwrite: bool = False,
) -> None:
    register("lattice", name, fn, description=description, overwrite=overwrite)


def register_defect(
    name: str,
    fn: Callable[..., Any],
    *,
    description: str = "",
    overwrite: bool = False,
) -> None:
    register("defect", name, fn, description=description, overwrite=overwrite)


def register_eval_task(
    name: str,
    fn: Callable[..., Any],
    *,
    description: str = "",
    overwrite: bool = False,
) -> None:
    register("eval_task", name, fn, description=description, overwrite=overwrite)
