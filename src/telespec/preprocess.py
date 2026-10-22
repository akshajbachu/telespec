"""Explicit preprocessing for irregular telemetry."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .io import sampling_report
from .models import FloatArray, Telemetry


def deduplicate(telemetry: Telemetry, *, method: str = "median") -> Telemetry:
    """Combine duplicate timestamps without silently dropping observations."""
    unique, inverse = np.unique(telemetry.timestamps, return_inverse=True)
    if unique.size == telemetry.timestamps.size:
        return telemetry
    values = np.empty(unique.size, dtype=float)
    for index in range(unique.size):
        group = telemetry.values[inverse == index]
        values[index] = np.median(group) if method == "median" else np.mean(group)
    return Telemetry(
        unique,
        values,
        telemetry.name,
        telemetry.unit,
        telemetry.source,
        sampling_report(unique),
    )


def log_transform(telemetry: Telemetry) -> Telemetry:
    """Apply signed log1p, preserving negative metrics if present."""
    values = np.sign(telemetry.values) * np.log1p(np.abs(telemetry.values))
    return Telemetry(
        telemetry.timestamps,
        values,
        telemetry.name,
        telemetry.unit,
        telemetry.source,
        telemetry.sampling,
    )


def robust_detrend(values: FloatArray, *, window: int = 31) -> tuple[FloatArray, FloatArray]:
    """Return median-filter trend and residual; window is forced odd."""
    y = np.asarray(values, dtype=float)
    size = max(3, min(int(window) | 1, (y.size - 1) | 1))
    trend = median_filter(y, size=size, mode="nearest")
    return np.asarray(trend, dtype=float), np.asarray(y - trend, dtype=float)


def resample_uniform(
    telemetry: Telemetry,
    *,
    interval: float | None = None,
    max_gap: float | None = None,
) -> Telemetry:
    """Explicitly linearly resample; raises when interpolation would bridge large gaps."""
    clean = deduplicate(telemetry)
    step = interval or (clean.sampling.median_interval if clean.sampling else None)
    if step is None or step <= 0:
        raise ValueError("resampling interval must be positive")
    gaps = np.diff(clean.timestamps)
    limit = max_gap if max_gap is not None else 5.0 * step
    if np.any(gaps > limit):
        raise ValueError("resampling would bridge a gap larger than max_gap")
    grid = np.arange(clean.timestamps[0], clean.timestamps[-1] + step * 0.5, step)
    values = np.interp(grid, clean.timestamps, clean.values)
    return Telemetry(
        grid,
        values,
        clean.name,
        clean.unit,
        clean.source,
        sampling_report(grid),
    )
