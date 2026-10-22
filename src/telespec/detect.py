"""Interpretable anomaly and change-point detectors."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.signal import welch
from sklearn.covariance import MinCovDet

from .models import Finding, FloatArray


@dataclass(frozen=True)
class ScoreSeries:
    times: FloatArray
    scores: FloatArray
    threshold: float
    features: FloatArray


def band_power_features(
    timestamps: FloatArray,
    values: FloatArray,
    *,
    window: int = 64,
    step: int | None = None,
    bands: int = 5,
) -> tuple[FloatArray, FloatArray]:
    """Build log octave-band powers plus residual variance by sliding window."""
    t = np.asarray(timestamps, dtype=float)
    y = np.asarray(values, dtype=float)
    size = min(max(8, int(window)), y.size)
    stride = step or max(1, size // 4)
    rows: list[FloatArray] = []
    centers: list[float] = []
    interval = float(np.median(np.diff(t)))
    for start in range(0, y.size - size + 1, stride):
        segment = y[start : start + size]
        frequencies, power = welch(
            segment,
            fs=1.0 / interval,
            nperseg=min(size, 64),
            detrend="linear",
        )
        positive = frequencies > 0
        frequency = frequencies[positive]
        density = power[positive]
        edges = np.geomspace(frequency[0], frequency[-1] * (1 + 1e-9), bands + 1)
        feature = []
        for low, high in pairwise(edges):
            mask = (frequency >= low) & (frequency < high)
            feature.append(float(np.log(np.trapezoid(density[mask], frequency[mask]) + 1e-12)))
        feature.append(float(np.log(np.var(segment) + 1e-12)))
        rows.append(np.asarray(feature, dtype=float))
        centers.append(float(t[start + size // 2]))
    if not rows:
        variance = float(np.log(np.var(y) + 1e-12))
        return np.asarray([float(np.median(t))]), np.asarray([[variance] * (bands + 1)])
    return np.asarray(centers), np.vstack(rows)


def robust_anomaly_scores(
    times: FloatArray,
    features: FloatArray,
    *,
    train_fraction: float = 0.3,
    false_positive_rate: float = 0.01,
) -> ScoreSeries:
    """Fit robust covariance on a prefix and calibrate scores empirically."""
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return ScoreSeries(
            np.asarray(times),
            np.zeros(matrix.shape[0]),
            float("inf"),
            matrix,
        )
    train_count = max(matrix.shape[1] + 2, int(np.ceil(matrix.shape[0] * train_fraction)))
    train_count = min(max(2, train_count), matrix.shape[0])
    train = matrix[:train_count]
    try:
        if train.shape[0] <= train.shape[1] + 1:
            raise ValueError("too few baseline windows for robust covariance")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimator = MinCovDet(random_state=0, support_fraction=0.75).fit(train)
            scores = np.asarray(estimator.mahalanobis(matrix), dtype=float)
    except (RuntimeWarning, UserWarning, ValueError):
        center = np.median(train, axis=0)
        scale = np.median(np.abs(train - center), axis=0) * 1.4826 + 1e-9
        scores = np.sum(np.square((matrix - center) / scale), axis=1)
    threshold = float(np.quantile(scores[:train_count], 1.0 - false_positive_rate))
    return ScoreSeries(np.asarray(times), scores, threshold, matrix)


def cusum_changes(
    timestamps: FloatArray,
    values: FloatArray,
    *,
    drift: float = 0.5,
    threshold: float = 5.0,
    baseline_fraction: float = 0.25,
) -> list[Finding]:
    """Detect persistent positive and negative mean shifts with two-sided CUSUM."""
    t = np.asarray(timestamps, dtype=float)
    y = np.asarray(values, dtype=float)
    baseline_count = min(y.size, max(5, round(y.size * baseline_fraction)))
    baseline = y[:baseline_count]
    center = float(np.median(baseline))
    scale = float(np.median(np.abs(baseline - center)) * 1.4826)
    if scale <= np.finfo(float).eps:
        return []
    standardized = (y - center) / scale
    positive = negative = 0.0
    findings: list[Finding] = []
    cooldown = 0
    for index, value in enumerate(standardized):
        positive = max(0.0, positive + value - drift)
        negative = min(0.0, negative + value + drift)
        if cooldown:
            cooldown -= 1
            continue
        statistic = max(positive, -negative)
        if statistic >= threshold:
            direction = 1.0 if positive >= -negative else -1.0
            summary = (
                f"Persistent level {'increase' if direction > 0 else 'decrease'} "
                f"detected near timestamp {t[index]:.3f}."
            )
            findings.append(
                Finding(
                    "level_shift",
                    float(t[max(0, index - 1)]),
                    float(t[min(t.size - 1, index + 1)]),
                    float(direction * statistic * scale),
                    float(1.0 - np.exp(-statistic / threshold)),
                    summary,
                    evidence={"cusum": statistic, "robust_scale": scale},
                )
            )
            positive = negative = 0.0
            cooldown = max(2, y.size // 20)
    return findings


def anomaly_findings(scores: ScoreSeries) -> list[Finding]:
    flagged = np.flatnonzero(scores.scores > scores.threshold)
    if not flagged.size:
        return []
    groups = np.split(flagged, np.flatnonzero(np.diff(flagged) > 1) + 1)
    findings: list[Finding] = []
    for group in groups:
        maximum = float(np.max(scores.scores[group]))
        findings.append(
            Finding(
                "outlier_interval",
                float(scores.times[group[0]]),
                float(scores.times[group[-1]]),
                maximum,
                float(np.clip(1.0 - scores.threshold / max(maximum, 1e-12), 0.0, 1.0)),
                (
                    "The telemetry spectrum and variance depart from the robust "
                    f"training baseline between {scores.times[group[0]]:.3f} and "
                    f"{scores.times[group[-1]]:.3f}."
                ),
                evidence={
                    "threshold": scores.threshold,
                    "peak_score": maximum,
                    "feature_vector": scores.features[
                        group[np.argmax(scores.scores[group])]
                    ].tolist(),
                },
            )
        )
    return findings


def sigma_baseline(values: FloatArray, *, z: float = 3.0) -> FloatArray:
    y = np.asarray(values, dtype=float)
    std = float(np.std(y))
    return np.abs(y - np.mean(y)) / std if std > 0 else np.zeros_like(y)


def ewma_baseline(values: FloatArray, *, alpha: float = 0.2) -> FloatArray:
    y = np.asarray(values, dtype=float)
    mean = np.empty_like(y)
    mean[0] = y[0]
    for index in range(1, y.size):
        mean[index] = alpha * y[index - 1] + (1 - alpha) * mean[index - 1]
    scale = float(np.std(y - mean) + 1e-12)
    return np.asarray(np.abs(y - mean) / scale, dtype=np.float64)


def rolling_robust_z(values: FloatArray, *, window: int = 31) -> FloatArray:
    y = np.asarray(values, dtype=float)
    output = np.zeros_like(y)
    for index in range(y.size):
        prior = y[max(0, index - window) : index]
        if prior.size < 3:
            continue
        center = np.median(prior)
        scale = np.median(np.abs(prior - center)) * 1.4826 + 1e-12
        output[index] = abs(y[index] - center) / scale
    return output
