import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from telespec.spectral import (
    Periodogram,
    baluev_fap,
    bootstrap_threshold,
    extract_peaks,
    frequency_grid,
    generalized_lomb_scargle,
    human_period,
    welch_periodogram,
)


def test_gls_recovers_irregular_sine():
    rng = np.random.default_rng(5)
    times = np.sort(rng.uniform(0, 3600, 180))
    truth = 1 / 300
    values = np.sin(2 * np.pi * truth * times) + rng.normal(0, 0.1, times.size)
    frequencies = np.linspace(1 / 800, 1 / 100, 300)
    result = generalized_lomb_scargle(times, values, frequencies)
    recovered = result.frequencies[np.argmax(result.power)]
    assert abs(recovered - truth) / truth < 0.03
    threshold, maxima = bootstrap_threshold(
        times, values, frequencies, iterations=20, random_state=1
    )
    peaks = extract_peaks(result, threshold, maxima)
    assert peaks and peaks[0].false_alarm_probability <= 1 / 21 + 1e-9


@given(st.floats(min_value=-1e5, max_value=1e5), st.floats(min_value=0.1, max_value=20))
@settings(max_examples=12)
def test_time_shift_and_amplitude_invariance(shift, amplitude):
    times = np.linspace(0, 100, 50)
    values = np.sin(times * 0.3)
    frequencies = np.linspace(0.01, 0.2, 30)
    first = generalized_lomb_scargle(times, values, frequencies).power
    second = generalized_lomb_scargle(times + shift, values * amplitude, frequencies).power
    assert np.allclose(first, second, atol=1e-8)


def test_constant_and_helpers():
    times = np.arange(20.0)
    result = generalized_lomb_scargle(times, np.ones(20), np.array([0.1, 0.2]))
    assert not np.any(result.power)
    assert frequency_grid(times).size >= 64
    assert welch_periodogram(np.sin(times), 1).frequencies.size
    assert 0 <= baluev_fap(0.8, 10) <= 1
    assert human_period(30) == "30 s"
    assert human_period(120).endswith("min")
    assert human_period(7200).endswith("h")
    assert human_period(172800).endswith("d")
    with pytest.raises(ValueError):
        bootstrap_threshold(times, times, np.array([0.1]), iterations=2)
    with pytest.raises(ValueError):
        generalized_lomb_scargle(times, times[:-1])


def test_harmonic_grouping():
    freq = np.array([0.09, 0.1, 0.11, 0.19, 0.2, 0.21])
    power = np.array([0.0, 0.9, 0.0, 0.0, 0.8, 0.0])
    peaks = extract_peaks(Periodogram(freq, power), 0.5, np.array([0.1, 0.2]))
    assert len(peaks) == 1
    assert peaks[0].harmonics == (0.2,)
