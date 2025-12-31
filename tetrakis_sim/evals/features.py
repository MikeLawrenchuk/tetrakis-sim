from __future__ import annotations

from typing import Any

import numpy as np

_TOPK = 5
_EPS = 1e-12


def spectral_features(freq: Any, amp: Any) -> dict[str, float]:
    f = np.asarray(freq, dtype=float)
    a = np.asarray(amp, dtype=float)

    if f.shape != a.shape:
        raise ValueError("freq and amp must have the same shape")

    # Keep non-negative frequencies
    mask = f >= 0
    f = f[mask]
    a = a[mask]

    s = float(a.sum())
    if a.size == 0 or s <= 0.0:
        out: dict[str, float] = {
            "dominant_freq": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "peak_amplitude": 0.0,
            "low_high_ratio": 0.0,
        }
        for k in range(_TOPK):
            out[f"peak_i_{k}"] = -1.0
            out[f"peak_mag_{k}"] = 0.0
        return out

    peak_idx = int(np.argmax(a))
    dominant = float(f[peak_idx])

    centroid = float((f * a).sum() / s)
    bandwidth = float(np.sqrt(((f - centroid) ** 2 * a).sum() / s))

    # Low/high energy ratio split by index (stable and fast)
    mid = int(a.size // 2)
    low = float(a[:mid].sum()) if mid > 0 else 0.0
    high = float(a[mid:].sum())
    low_high_ratio = float(low / (high + _EPS))

    # Top-k peaks by amplitude: store bin index + normalized magnitude
    k = min(_TOPK, int(a.size))
    idx = np.argpartition(a, -k)[-k:]
    idx = idx[np.argsort(a[idx])[::-1]]

    amax = float(a.max())
    out = {
        "dominant_freq": dominant,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "peak_amplitude": amax,
        "low_high_ratio": low_high_ratio,
    }
    for j in range(_TOPK):
        if j < idx.size:
            i = int(idx[j])
            out[f"peak_i_{j}"] = float(i)
            out[f"peak_mag_{j}"] = float(a[i] / (amax + _EPS))
        else:
            out[f"peak_i_{j}"] = -1.0
            out[f"peak_mag_{j}"] = 0.0

    return out


def timeseries_features(values: Any) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"ts_rms": 0.0, "ts_peak": 0.0}
    return {
        "ts_rms": float(np.sqrt(np.mean(v**2))),
        "ts_peak": float(np.max(np.abs(v))),
    }
