import json

import numpy as np
import pytest

from telespec.io import parse_timestamp, read_csv, read_jsonl, sampling_report
from telespec.preprocess import deduplicate, log_transform, resample_uniform, robust_detrend


def test_csv_sampling_report_and_epoch_milliseconds():
    data = "timestamp,latency\n1700000000000,10\n1700000060000,12\n1700000240000,15\n"
    telemetry = read_csv(data, value="latency")
    assert telemetry.timestamps[0] == 1_700_000_000
    assert telemetry.sampling is not None
    assert telemetry.sampling.gaps
    assert telemetry.sampling.coverage < 1


def test_jsonl_iso_timestamp_and_errors():
    rows = [{"time": f"2024-01-01T00:0{i}:00Z", "v": i} for i in range(3)]
    telemetry = read_jsonl("\n".join(json.dumps(row) for row in rows), timestamp="time", value="v")
    assert telemetry.values.tolist() == [0, 1, 2]
    assert parse_timestamp("2024-01-01T00:00:00Z") > 0
    with pytest.raises(ValueError, match="invalid JSON"):
        read_jsonl("{}\n{")
    with pytest.raises(ValueError, match="invalid timestamp"):
        parse_timestamp("yesterday")


def test_deduplicate_transform_and_resample():
    telemetry = read_csv("timestamp,value\n0,1\n1,2\n1,4\n2,8\n")
    clean = deduplicate(telemetry)
    assert clean.values.tolist() == [1, 3, 8]
    assert np.allclose(log_transform(clean).values, np.log1p(clean.values))
    regular = resample_uniform(clean, interval=0.5)
    assert regular.values.size == 5
    trend, residual = robust_detrend(regular.values, window=3)
    assert np.allclose(trend + residual, regular.values)


def test_resample_rejects_large_gap():
    telemetry = read_csv("timestamp,value\n0,1\n1,2\n20,3\n")
    with pytest.raises(ValueError, match="bridge"):
        resample_uniform(telemetry, interval=1, max_gap=3)
    with pytest.raises(ValueError, match="distinct"):
        sampling_report(np.array([1.0, 1.0, 1.0]))
