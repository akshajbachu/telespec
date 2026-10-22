from telespec.analysis import analyze
from telespec.validation import Injection, generate


def test_demo_regression_shape_and_peak():
    telemetry = generate(
        points=120,
        interval=60,
        daily_amplitude=0,
        noise=0.2,
        injections=(Injection("periodicity", 0, 1, 6, 900),),
        random_state=12,
    ).telemetry
    result = analyze(telemetry, bootstrap_iterations=20, random_state=12)
    assert result.peaks
    assert abs(result.peaks[0].period - 900) / 900 < 0.08
    assert result.peaks[0].power > 0.8
