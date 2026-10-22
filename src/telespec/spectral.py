"""Spectral estimators for irregular and regular telemetry.

The generalized Lomb-Scargle implementation follows Zechmeister & Kuerster
(2009), A&A 496, 577. It fits ``y = a cos(wt) + b sin(wt) + c`` independently
at every frequency and reports the fractional chi-square improvement over a
floating-mean model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, welch

from .models import FloatArray, SpectralPeak


@dataclass(frozen=True)
class Periodogram:
    frequencies: FloatArray
    power: FloatArray


def frequency_grid(
    timestamps: FloatArray,
    *,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    samples_per_peak: int = 8,
) -> FloatArray:
    """Construct an oversampled frequency grid from the observation baseline."""
    times = np.sort(np.unique(np.asarray(timestamps, dtype=float)))
    baseline = float(np.ptp(times))
    if times.size < 3 or baseline <= 0:
        raise ValueError("at least three distinct timestamps are required")
    positive = np.diff(times)
    low = minimum_frequency or 1.0 / baseline
    high = maximum_frequency or 0.5 / float(np.median(positive))
    if low <= 0 or high <= low:
        raise ValueError("frequency bounds must satisfy 0 < minimum < maximum")
    requested = int(np.ceil((high - low) * baseline * samples_per_peak))
    count = min(4096, max(64, requested))
    return np.linspace(low, high, count, dtype=float)


def generalized_lomb_scargle(
    timestamps: FloatArray,
    values: FloatArray,
    frequencies: FloatArray | None = None,
    *,
    weights: FloatArray | None = None,
) -> Periodogram:
    """Compute a floating-mean generalized Lomb-Scargle periodogram.

    The direct least-squares formulation is intentionally transparent and
    stable for the dataset sizes targeted by telespec.
    """
    t = np.asarray(timestamps, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape != y.shape or t.ndim != 1 or t.size < 3:
        raise ValueError("timestamps and values must be equal one-dimensional arrays")
    order = np.argsort(t)
    t = t[order] - np.min(t)
    y = y[order]
    w = np.ones(y.size, dtype=float) if weights is None else np.asarray(weights, dtype=float)[order]
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and positive")
    w = w / np.sum(w)
    freq = frequency_grid(t) if frequencies is None else np.asarray(frequencies, dtype=float)
    if np.any(freq <= 0):
        raise ValueError("frequencies must be positive")
    mean = float(np.dot(w, y))
    chi0 = float(np.dot(w, np.square(y - mean)))
    if chi0 <= np.finfo(float).eps:
        return Periodogram(freq, np.zeros(freq.size, dtype=float))
    root_w = np.sqrt(w)
    power = np.empty(freq.size, dtype=float)
    for index, candidate in enumerate(freq):
        phase = 2.0 * np.pi * candidate * t
        design = np.column_stack((np.cos(phase), np.sin(phase), np.ones(t.size)))
        coefficients, *_ = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)
        residual = y - design @ coefficients
        chi = float(np.dot(w, np.square(residual)))
        power[index] = np.clip(1.0 - chi / chi0, 0.0, 1.0)
    return Periodogram(freq, power)


def bootstrap_threshold(
    timestamps: FloatArray,
    values: FloatArray,
    frequencies: FloatArray,
    *,
    false_alarm_probability: float = 0.01,
    iterations: int = 200,
    random_state: int | np.random.Generator | None = None,
) -> tuple[float, FloatArray]:
    """Estimate a global maximum-power threshold by permutation bootstrap."""
    if not 0 < false_alarm_probability < 1:
        raise ValueError("false_alarm_probability must lie between zero and one")
    if iterations < 10:
        raise ValueError("iterations must be at least 10")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    maxima = np.empty(iterations, dtype=float)
    y = np.asarray(values, dtype=float)
    for index in range(iterations):
        maxima[index] = np.max(
            generalized_lomb_scargle(timestamps, rng.permutation(y), frequencies).power
        )
    return float(np.quantile(maxima, 1.0 - false_alarm_probability)), np.sort(maxima)


def empirical_fap(power: float, bootstrap_maxima: FloatArray) -> float:
    maxima = np.asarray(bootstrap_maxima, dtype=float)
    return float((1 + np.count_nonzero(maxima >= power)) / (maxima.size + 1))


def baluev_fap(power: float, independent_frequencies: int) -> float:
    """Fast independent-frequency approximation used as a cross-check."""
    if independent_frequencies < 1:
        raise ValueError("independent_frequencies must be positive")
    single = max(0.0, 1.0 - float(power))
    return float(np.clip(1.0 - (1.0 - single) ** independent_frequencies, 0.0, 1.0))


def extract_peaks(
    periodogram: Periodogram,
    threshold: float,
    bootstrap_maxima: FloatArray,
    *,
    harmonic_tolerance: float = 0.04,
) -> list[SpectralPeak]:
    """Extract significant local maxima and merge integer harmonics."""
    indexes, _properties = find_peaks(periodogram.power, height=threshold)
    candidates = sorted(indexes, key=lambda i: periodogram.power[i], reverse=True)
    peaks: list[SpectralPeak] = []
    for index in candidates:
        frequency = float(periodogram.frequencies[index])
        harmonic_of: int | None = None
        for parent_index, parent in enumerate(peaks):
            ratio = frequency / parent.frequency
            nearest = round(ratio)
            if nearest >= 2 and abs(ratio - nearest) <= harmonic_tolerance * nearest:
                harmonic_of = parent_index
                break
        if harmonic_of is not None:
            parent = peaks[harmonic_of]
            peaks[harmonic_of] = SpectralPeak(
                parent.frequency,
                parent.period,
                parent.power,
                parent.false_alarm_probability,
                (*parent.harmonics, frequency),
            )
            continue
        power = float(periodogram.power[index])
        peaks.append(
            SpectralPeak(
                frequency,
                1.0 / frequency,
                power,
                empirical_fap(power, bootstrap_maxima),
            )
        )
    return sorted(peaks, key=lambda peak: peak.power, reverse=True)


def welch_periodogram(values: FloatArray, sample_interval: float) -> Periodogram:
    """Welch PSD for explicitly regular data."""
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    y = np.asarray(values, dtype=float)
    frequencies, power = welch(
        y,
        fs=1.0 / sample_interval,
        nperseg=min(256, y.size),
        detrend="linear",
        scaling="density",
    )
    keep = frequencies > 0
    return Periodogram(np.asarray(frequencies[keep]), np.asarray(power[keep]))


def human_period(seconds: float) -> str:
    if seconds >= 86400:
        return f"{seconds / 86400:.2g} d"
    if seconds >= 3600:
        return f"{seconds / 3600:.2g} h"
    if seconds >= 60:
        return f"{seconds / 60:.2g} min"
    return f"{seconds:.2g} s"
