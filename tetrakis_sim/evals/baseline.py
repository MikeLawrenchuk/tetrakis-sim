from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _feature_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        keys.update((r.get("features") or {}).keys())
    return sorted(keys)


def _vectorize(rows: list[dict[str, Any]], keys: list[str]) -> tuple[np.ndarray, list[str]]:
    X = np.zeros((len(rows), len(keys)), dtype=float)
    y: list[str] = []
    for i, r in enumerate(rows):
        y.append(str(r.get("label")))
        feats = r.get("features") or {}
        for j, k in enumerate(keys):
            X[i, j] = float(feats.get(k, 0.0))
    return X, y


def _split(n: int, *, seed: int, test_frac: float) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = max(1, int(math.floor(test_frac * n)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if not train_idx:
        train_idx = test_idx
    return train_idx, test_idx


def run_nearest_centroid_baseline(
    data_path: str | Path,
    *,
    seed: int = 0,
    test_frac: float = 0.3,
) -> dict[str, float]:
    rows = _load_jsonl(data_path)
    if not rows:
        raise ValueError("Dataset is empty")

    keys = _feature_keys(rows)
    X, y = _vectorize(rows, keys)

    train_idx, test_idx = _split(len(rows), seed=seed, test_frac=test_frac)
    Xtr = X[train_idx]
    ytr = [y[i] for i in train_idx]
    Xte = X[test_idx]
    yte = [y[i] for i in test_idx]

    labels = sorted(set(ytr))
    centroids: dict[str, np.ndarray] = {}
    for lab in labels:
        mask = np.array([yy == lab for yy in ytr], dtype=bool)
        centroids[lab] = Xtr[mask].mean(axis=0) if mask.any() else Xtr.mean(axis=0)

    correct = 0
    for i in range(len(Xte)):
        x = Xte[i]
        best_lab = None
        best_d = None
        for lab, c in centroids.items():
            d = float(np.linalg.norm(x - c))
            if best_d is None or d < best_d:
                best_d = d
                best_lab = lab
        if best_lab == yte[i]:
            correct += 1

    acc = float(correct / max(1, len(Xte)))

    return {
        "n": float(len(rows)),
        "n_train": float(len(train_idx)),
        "n_test": float(len(test_idx)),
        "accuracy": acc,
    }
