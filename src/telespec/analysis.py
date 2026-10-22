"""High-level, reproducible telespec analysis pipeline."""

from __future__ import annotations

import numpy as np

from .decompose import decompose
from .detect import anomaly_findings, band_power_features, cusum_changes, robust_anomaly_scores
from .io import sampling_report
from .models import AnalysisResult, Finding, Telemetry
from .preprocess import deduplicate
from .spectral import (
    bootstrap_threshold,
    extract_peaks,
    generalized_lomb_scargle,
    human_period,
)
from .timefreq import stft_spectrogram


def _longest_gap_safe_segment(telemetry: Telemetry) -> Telemetry:
    """Select the longest run that can be resampled without bridging a large gap."""
    segment = telemetry
    while True:
        report = segment.sampling or sampling_report(segment.timestamps)
        split_points = np.flatnonzero(np.diff(segment.timestamps) > 5 * report.median_interval) + 1
        if not split_points.size:
            return segment
        groups = np.split(np.arange(segment.values.size), split_points)
        group = max(groups, key=len)
        if group.size < 3:
            raise ValueError("no gap-safe segment contains at least three observations")
        times = segment.timestamps[group]
        segment = Telemetry(
            times,
            segment.values[group],
            segment.name,
            segment.unit,
            segment.source,
            sampling_report(times),
        )


def analyze(
    telemetry: Telemetry,
    *,
    false_alarm_probability: float = 0.01,
    bootstrap_iterations: int = 100,
    train_fraction: float = 0.3,
    random_state: int | None = 0,
) -> AnalysisResult:
    """Run spectral estimation, decomposition, and interpretable detection."""
    clean = deduplicate(telemetry)
    periodogram = generalized_lomb_scargle(clean.timestamps, clean.values)
    threshold, maxima = bootstrap_threshold(
        clean.timestamps,
        clean.values,
        periodogram.frequencies,
        false_alarm_probability=false_alarm_probability,
        iterations=bootstrap_iterations,
        random_state=random_state,
    )
    peaks = extract_peaks(periodogram, threshold, maxima)
    dominant_period = peaks[0].period if peaks else None
    segment = _longest_gap_safe_segment(clean)
    decomposition = decompose(segment, period_seconds=dominant_period)
    spectrogram = stft_spectrogram(segment, window=min(128, segment.values.size))
    feature_times, features = band_power_features(
        decomposition.timestamps,
        decomposition.remainder,
    )
    scores = robust_anomaly_scores(
        feature_times,
        features,
        train_fraction=train_fraction,
        false_positive_rate=false_alarm_probability,
    )
    findings: list[Finding] = [
        Finding(
            "periodicity",
            float(clean.timestamps[0]),
            float(clean.timestamps[-1]),
            peak.power,
            1.0 - peak.false_alarm_probability,
            (
                f"Significant {human_period(peak.period)} periodicity "
                f"(bootstrap FAP {peak.false_alarm_probability:.3g})."
            ),
            peak.period,
            peak.frequency,
            {"power": peak.power, "harmonics": list(peak.harmonics)},
        )
        for peak in peaks
    ]
    findings.extend(cusum_changes(decomposition.timestamps, decomposition.remainder))
    findings.extend(anomaly_findings(scores))
    findings.sort(key=lambda finding: (finding.start, finding.kind))
    return AnalysisResult(
        clean,
        periodogram.frequencies,
        periodogram.power,
        threshold,
        peaks,
        findings,
        decomposition.trend,
        decomposition.seasonal,
        decomposition.remainder,
        scores.times,
        scores.scores,
        {
            "false_alarm_probability": false_alarm_probability,
            "bootstrap_iterations": bootstrap_iterations,
            "train_fraction": train_fraction,
            "resampling": "linear interpolation only for STL and sliding-window features",
            "analysis_segment": {
                "start": float(segment.timestamps[0]),
                "end": float(segment.timestamps[-1]),
                "observations": int(segment.values.size),
            },
            "spectrogram": {
                "times": spectrogram.times.tolist(),
                "frequencies": spectrogram.frequencies.tolist(),
                "power": spectrogram.power.tolist(),
            },
        },
    )
