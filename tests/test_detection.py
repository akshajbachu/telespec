import numpy as np

from telespec.decompose import decompose
from telespec.detect import (
    anomaly_findings,
    band_power_features,
    cusum_changes,
    ewma_baseline,
    robust_anomaly_scores,
    rolling_robust_z,
    sigma_baseline,
)
from telespec.timefreq import morlet_scalogram, ridge_changes, spectral_ridge, stft_spectrogram
from telespec.validation import Injection, generate


def test_cusum_localizes_shift_and_constant_is_quiet():
    times = np.arange(200.0)
    rng = np.random.default_rng(2)
    values = rng.normal(0, 0.2, 200)
    values[100:] += 3
    findings = cusum_changes(times, values)
    assert findings
    assert 90 <= findings[0].start <= 120
    assert cusum_changes(times, np.ones(200)) == []


def test_features_robust_scoring_and_baselines():
    rng = np.random.default_rng(3)
    times = np.arange(300.0)
    values = rng.normal(size=300)
    values[230:250] *= 8
    centers, features = band_power_features(times, values, window=48)
    scores = robust_anomaly_scores(centers, features)
    assert scores.scores.shape == centers.shape
    assert anomaly_findings(scores)
    assert sigma_baseline(values).shape == values.shape
    assert ewma_baseline(values).shape == values.shape
    assert rolling_robust_z(values).shape == values.shape


def test_decomposition_and_time_frequency():
    sample = generate(
        points=180,
        interval=10,
        daily_amplitude=0,
        noise=0.1,
        injections=(Injection("periodicity", 0, 1, 5, 120),),
    ).telemetry
    decomposition = decompose(sample, period_seconds=120)
    assert decomposition.period_samples == 12
    assert np.var(decomposition.remainder) < np.var(sample.values)
    stft = stft_spectrogram(sample, window=32)
    wavelet = morlet_scalogram(sample)
    frequencies, powers = spectral_ridge(wavelet)
    assert stft.power.ndim == wavelet.power.ndim == 2
    assert frequencies.shape == powers.shape == wavelet.times.shape
    assert ridge_changes(wavelet).ndim == 1
