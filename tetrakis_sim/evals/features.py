from __future__ import annotations

from typing import Any

import numpy as np


def spectral_features(freq: Any, amp: Any) -> dict[str, float]:
    f = np.asarray(freq, dtype=float)
    a = np.asarray(amp, dtype=float)

    if f.shape != a.shape:
        raise ValueError("freq and amp must have the same shape")

    mask = f >= 0
    f = np.abs(f[mask])
    a = a[mask]

    s = float(a.sum())
    if a.size == 0 or s <= 0.0:
        return {
            "dominant_freq": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "peak_amplitude": 0.0,
        }

    peak_idx = int(np.argmax(a))
    dominant = float(f[peak_idx])

    centroid = float((f * a).sum() / s)
    bandwidth = float(np.sqrt(((f - centroid) ** 2 * a).sum() / s))

    return {
        "dominant_freq": dominant,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "peak_amplitude": float(a.max()),
    }


def timeseries_features(values: Any) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"ts_rms": 0.0, "ts_peak": 0.0}
    return {
        "ts_rms": float(np.sqrt(np.mean(v**2))),
        "ts_peak": float(np.max(np.abs(v))),
    }
