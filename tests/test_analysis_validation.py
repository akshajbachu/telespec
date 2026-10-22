import json

import numpy as np

from telespec.analysis import analyze
from telespec.findings import findings_json, findings_markdown, write_report
from telespec.models import Telemetry
from telespec.validation import Injection, generate


def test_synthetic_generator_and_full_analysis(tmp_path):
    synthetic = generate(
        points=180,
        interval=60,
        daily_amplitude=0,
        missing=0.1,
        jitter=0.05,
        injections=(
            Injection("periodicity", 0, 1, 7, 900),
            Injection("level_shift", 0.7, 0.9, 9),
            Injection("variance_burst", 0.4, 0.5, 4),
            Injection("spikes", 0.2, 0.3, 12),
        ),
        random_state=4,
    )
    result = analyze(synthetic.telemetry, bootstrap_iterations=20)
    assert result.peaks
    payload = result.to_dict()
    assert payload["findings"] and payload["frequencies"]
    assert json.loads(findings_json(result.findings))
    assert "# telespec" in findings_markdown(result.findings)
    assert write_report(result.findings, tmp_path / "report.md").exists()
    assert write_report(result.findings, tmp_path / "report.json").exists()


def test_models_reject_invalid_data():
    try:
        Telemetry(np.arange(2.0), np.arange(2.0))
    except ValueError as error:
        assert "three" in str(error)
    else:
        raise AssertionError("expected ValueError")
