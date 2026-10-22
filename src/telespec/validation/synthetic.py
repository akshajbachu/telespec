"""Reproducible network-like telemetry with exact anomaly ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..io import sampling_report
from ..models import Telemetry

InjectionKind = Literal["periodicity", "level_shift", "variance_burst", "spikes"]


@dataclass(frozen=True)
class Injection:
    kind: InjectionKind
    start_fraction: float = 0.6
    end_fraction: float = 0.8
    magnitude: float = 3.0
    period: float = 300.0


@dataclass(frozen=True)
class SyntheticTelemetry:
    telemetry: Telemetry
    injections: tuple[Injection, ...]


def generate(
    *,
    points: int = 720,
    interval: float = 60.0,
    baseline: float = 50.0,
    daily_amplitude: float = 5.0,
    noise: float = 1.0,
    jitter: float = 0.0,
    missing: float = 0.0,
    injections: tuple[Injection, ...] = (),
    random_state: int | None = 0,
) -> SyntheticTelemetry:
    """Generate heavy-tailed telemetry and apply parameterized effects."""
    if points < 20 or interval <= 0 or not 0 <= missing < 0.9:
        raise ValueError("invalid generator parameters")
    rng = np.random.default_rng(random_state)
    times = np.arange(points, dtype=float) * interval
    if jitter:
        times += rng.normal(0.0, jitter * interval, points)
        times = np.sort(times)
    values = (
        baseline
        + daily_amplitude * np.sin(2 * np.pi * times / 86400.0)
        + rng.standard_t(df=5, size=points) * noise
    )
    for injection in injections:
        start = max(0, min(points - 1, int(points * injection.start_fraction)))
        end = max(start + 1, min(points, int(points * injection.end_fraction)))
        if injection.kind == "periodicity":
            values[start:end] += injection.magnitude * np.sin(
                2 * np.pi * times[start:end] / injection.period
            )
        elif injection.kind == "level_shift":
            values[start:end] += injection.magnitude
        elif injection.kind == "variance_burst":
            values[start:end] += rng.normal(0, injection.magnitude, end - start)
        elif injection.kind == "spikes":
            count = max(1, (end - start) // 20)
            indexes = rng.choice(np.arange(start, end), size=count, replace=False)
            values[indexes] += injection.magnitude
    keep = rng.random(points) >= missing
    telemetry = Telemetry(
        times[keep],
        values[keep],
        "latency",
        "ms",
        "synthetic",
        sampling_report(times[keep]),
    )
    return SyntheticTelemetry(telemetry, injections)
