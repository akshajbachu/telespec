"""Seasonal-trend decomposition with defensible fallbacks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.seasonal import STL

from .models import FloatArray, Telemetry
from .preprocess import resample_uniform, robust_detrend


@dataclass(frozen=True)
class Decomposition:
    timestamps: FloatArray
    trend: FloatArray
    seasonal: FloatArray
    remainder: FloatArray
    period_samples: int | None


def decompose(telemetry: Telemetry, *, period_seconds: float | None = None) -> Decomposition:
    regular = resample_uniform(telemetry)
    interval = regular.sampling.median_interval if regular.sampling else 1.0
    period = round(period_seconds / interval) if period_seconds else 0
    if period >= 3 and regular.values.size >= 2 * period:
        result = STL(regular.values, period=period, robust=True).fit()
        return Decomposition(
            regular.timestamps,
            np.asarray(result.trend, dtype=float),
            np.asarray(result.seasonal, dtype=float),
            np.asarray(result.resid, dtype=float),
            period,
        )
    trend, remainder = robust_detrend(regular.values, window=max(3, regular.values.size // 10))
    return Decomposition(
        regular.timestamps,
        trend,
        np.zeros(regular.values.size, dtype=float),
        remainder,
        None,
    )
