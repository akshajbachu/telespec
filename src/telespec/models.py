"""Typed records shared across the analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Gap:
    start: float
    end: float
    missing_intervals: int


@dataclass(frozen=True)
class SamplingReport:
    count: int
    start: float
    end: float
    median_interval: float
    jitter_mad: float
    coverage: float
    duplicate_count: int
    gaps: tuple[Gap, ...] = ()

    @property
    def is_regular(self) -> bool:
        return self.jitter_mad <= max(self.median_interval * 0.01, 1e-12) and not self.gaps


@dataclass(frozen=True)
class Telemetry:
    timestamps: FloatArray
    values: FloatArray
    name: str = "value"
    unit: str = ""
    source: str = ""
    sampling: SamplingReport | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.timestamps, dtype=float)
        y = np.asarray(self.values, dtype=float)
        if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
            raise ValueError("timestamps and values must be equal-length one-dimensional arrays")
        if t.size < 3:
            raise ValueError("telemetry requires at least three observations")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
            raise ValueError("telemetry contains non-finite observations")
        object.__setattr__(self, "timestamps", t)
        object.__setattr__(self, "values", y)


FindingKind = Literal[
    "periodicity",
    "periodicity_change",
    "level_shift",
    "variance_change",
    "outlier_interval",
]


@dataclass(frozen=True)
class Finding:
    kind: FindingKind
    start: float
    end: float
    magnitude: float
    confidence: float
    summary: str
    period: float | None = None
    frequency: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpectralPeak:
    frequency: float
    period: float
    power: float
    false_alarm_probability: float
    harmonics: tuple[float, ...] = ()


@dataclass
class AnalysisResult:
    telemetry: Telemetry
    frequencies: FloatArray
    power: FloatArray
    threshold_power: float
    peaks: list[SpectralPeak]
    findings: list[Finding]
    trend: FloatArray
    seasonal: FloatArray
    remainder: FloatArray
    anomaly_times: FloatArray
    anomaly_scores: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_series: bool = True) -> dict[str, Any]:
        sampling = asdict(self.telemetry.sampling) if self.telemetry.sampling else None
        result: dict[str, Any] = {
            "name": self.telemetry.name,
            "unit": self.telemetry.unit,
            "source": self.telemetry.source,
            "sampling": sampling,
            "peaks": [asdict(peak) for peak in self.peaks],
            "findings": [finding.to_dict() for finding in self.findings],
            "threshold_power": self.threshold_power,
            "metadata": self.metadata,
        }
        if include_series:
            result.update(
                {
                    "timestamps": self.telemetry.timestamps.tolist(),
                    "values": self.telemetry.values.tolist(),
                    "frequencies": self.frequencies.tolist(),
                    "power": self.power.tolist(),
                    "trend": self.trend.tolist(),
                    "seasonal": self.seasonal.tolist(),
                    "remainder": self.remainder.tolist(),
                    "anomaly_times": self.anomaly_times.tolist(),
                    "anomaly_scores": self.anomaly_scores.tolist(),
                }
            )
        return result
