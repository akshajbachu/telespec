"""Time-frequency analysis on explicitly regularized telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, stft

from .models import FloatArray, Telemetry
from .preprocess import resample_uniform


@dataclass(frozen=True)
class TimeFrequency:
    times: FloatArray
    frequencies: FloatArray
    power: FloatArray
    method: str


def stft_spectrogram(
    telemetry: Telemetry,
    *,
    window: int = 64,
    overlap: float = 0.75,
) -> TimeFrequency:
    regular = resample_uniform(telemetry)
    step = regular.sampling.median_interval if regular.sampling else 1.0
    segment = min(max(8, window), regular.values.size)
    frequencies, times, coefficients = stft(
        regular.values,
        fs=1.0 / step,
        nperseg=segment,
        noverlap=min(segment - 1, int(segment * overlap)),
        detrend="linear",
    )
    keep = frequencies > 0
    return TimeFrequency(
        regular.timestamps[0] + np.asarray(times),
        np.asarray(frequencies[keep]),
        np.square(np.abs(coefficients[keep])),
        "stft",
    )


def morlet_scalogram(
    telemetry: Telemetry,
    *,
    periods: FloatArray | None = None,
    omega0: float = 6.0,
) -> TimeFrequency:
    regular = resample_uniform(telemetry)
    step = regular.sampling.median_interval if regular.sampling else 1.0
    available = float(np.ptp(regular.timestamps))
    candidate_periods = (
        np.geomspace(max(2 * step, step), max(4 * step, available / 3), 48)
        if periods is None
        else np.asarray(periods, dtype=float)
    )
    widths = candidate_periods * omega0 / (2 * np.pi * step)
    centered = regular.values - np.mean(regular.values)
    coefficients = np.empty((widths.size, centered.size), dtype=np.complex128)
    for index, width in enumerate(widths):
        radius = max(4, int(np.ceil(5 * width)))
        coordinates = np.arange(-radius, radius + 1, dtype=float)
        wavelet = (
            np.pi**-0.25
            * np.exp(1j * omega0 * coordinates / width)
            * np.exp(-0.5 * np.square(coordinates / width))
            / np.sqrt(width)
        )
        coefficients[index] = fftconvolve(centered, np.conjugate(wavelet[::-1]), mode="same")
    return TimeFrequency(
        regular.timestamps,
        1.0 / candidate_periods,
        np.square(np.abs(coefficients)),
        "morlet",
    )


def spectral_ridge(time_frequency: TimeFrequency) -> tuple[FloatArray, FloatArray]:
    """Return dominant frequency and power at every time bin."""
    indexes = np.argmax(time_frequency.power, axis=0)
    columns = np.arange(indexes.size)
    return time_frequency.frequencies[indexes], time_frequency.power[indexes, columns]


def ridge_changes(
    time_frequency: TimeFrequency,
    *,
    relative_jump: float = 0.35,
) -> FloatArray:
    frequencies, _ = spectral_ridge(time_frequency)
    denominator = np.maximum(frequencies[:-1], np.finfo(float).eps)
    jumps = np.abs(np.diff(frequencies)) / denominator
    return time_frequency.times[1:][jumps >= relative_jump]
