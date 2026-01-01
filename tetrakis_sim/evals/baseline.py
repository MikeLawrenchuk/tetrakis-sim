from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np

SplitMode = Literal["random", "stratified"]


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


def _split_random(n: int, *, seed: int, test_frac: float) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = max(1, int(math.floor(test_frac * n)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if not train_idx:
        train_idx = test_idx
    return train_idx, test_idx


def _split_stratified(
    y: list[str],
    *,
    seed: int,
    test_frac: float,
) -> tuple[list[int], list[int]]:
    """Stratified split by label.

    When a label has >=2 examples, attempts to put at least one example in both
    train and test for that label. Singletons are used to meet the desired test size.
    """
    n = len(y)
    if n <= 1:
        return [0], [0]

    rng = random.Random(seed)

    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(y):
        by_label.setdefault(lab, []).append(i)

    for idxs in by_label.values():
        rng.shuffle(idxs)

    desired_test = max(1, int(math.floor(test_frac * n)))

    train_idx: list[int] = []
    test_idx: list[int] = []
    singles: list[int] = []

    # Allocate multi-example labels first
    for idxs in by_label.values():
        if len(idxs) == 1:
            singles.append(idxs[0])
            continue

        n_lab = len(idxs)
        n_test_lab = int(math.floor(test_frac * n_lab))
        n_test_lab = max(1, n_test_lab)  # ensure test has something
        n_test_lab = min(n_test_lab, n_lab - 1)  # ensure train has something

        test_idx.extend(idxs[:n_test_lab])
        train_idx.extend(idxs[n_test_lab:])

    # Allocate singletons to meet desired test size
    rng.shuffle(singles)
    remaining = desired_test - len(test_idx)
    if remaining > 0:
        test_idx.extend(singles[:remaining])
        train_idx.extend(singles[remaining:])
    else:
        train_idx.extend(singles)

    # Guardrails
    if not test_idx and train_idx:
        test_idx = [train_idx.pop(0)]
    if not train_idx:
        train_idx = test_idx[:]  # degenerate fallback

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def run_nearest_centroid_baseline(
    data_path: str | Path,
    *,
    seed: int = 0,
    test_frac: float = 0.3,
    split: SplitMode = "stratified",
) -> dict[str, float]:
    rows = _load_jsonl(data_path)
    if not rows:
        raise ValueError("Dataset is empty")

    keys = _feature_keys(rows)
    X, y = _vectorize(rows, keys)

    if split == "random":
        train_idx, test_idx = _split_random(len(rows), seed=seed, test_frac=test_frac)
    elif split == "stratified":
        train_idx, test_idx = _split_stratified(y, seed=seed, test_frac=test_frac)
    else:
        raise ValueError(f"Unknown split mode: {split}")

    Xtr = X[train_idx]
    ytr = [y[i] for i in train_idx]
    Xte = X[test_idx]
    yte = [y[i] for i in test_idx]

    # Standardize using training statistics (z-score)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0.0] = 1.0

    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

    labels = sorted(set(ytr))
    centroids: dict[str, np.ndarray] = {}
    for lab in labels:
        mask = np.array([yy == lab for yy in ytr], dtype=bool)
        centroids[lab] = Xtr[mask].mean(axis=0) if mask.any() else Xtr.mean(axis=0)

    correct = 0
    preds: list[str] = []
    for i in range(len(Xte)):
        x = Xte[i]
        best_lab = None
        best_d = None
        for lab, c in centroids.items():
            d = float(np.linalg.norm(x - c))
            if best_d is None or d < best_d:
                best_d = d
                best_lab = lab
        pred = str(best_lab)
        preds.append(pred)
        if pred == yte[i]:
            correct += 1

    acc = float(correct / max(1, len(Xte)))

    # Macro-F1
    f1s: list[float] = []
    for lab in sorted(set(yte) | set(preds)):
        tp = sum(1 for p, t in zip(preds, yte) if p == lab and t == lab)
        fp = sum(1 for p, t in zip(preds, yte) if p == lab and t != lab)
        fn = sum(1 for p, t in zip(preds, yte) if p != lab and t == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(float(f1))
    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0

    return {
        "n": float(len(rows)),
        "n_train": float(len(train_idx)),
        "n_test": float(len(test_idx)),
        "accuracy": acc,
        "macro_f1": macro_f1,
    }
