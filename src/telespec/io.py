"""CSV and JSONL ingestion with explicit sampling diagnostics."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .models import FloatArray, Gap, SamplingReport, Telemetry


def parse_timestamp(value: Any) -> float:
    """Parse ISO-8601 or epoch seconds/milliseconds into epoch seconds."""
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        try:
            number = float(text)
        except ValueError:
            normalized = text.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(normalized).timestamp()
            except ValueError as error:
                raise ValueError(f"invalid timestamp: {value!r}") from error
    return number / 1000.0 if abs(number) > 1e11 else number


def sampling_report(timestamps: FloatArray, *, gap_factor: float = 3.0) -> SamplingReport:
    times = np.sort(np.asarray(timestamps, dtype=float))
    if times.size < 2:
        raise ValueError("at least two timestamps are required")
    diffs = np.diff(times)
    duplicate_count = int(np.count_nonzero(diffs == 0))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        raise ValueError("timestamps must contain at least two distinct values")
    # A lower median prevents one large gap from inflating the cadence estimate
    # when only a few observations are available.
    interval = float(np.quantile(positive, 0.5, method="lower"))
    jitter = float(np.median(np.abs(positive - interval)))
    unique = np.unique(times)
    unique_diffs = np.diff(unique)
    gaps = tuple(
        Gap(
            float(unique[i]),
            float(unique[i + 1]),
            max(1, round(unique_diffs[i] / interval) - 1),
        )
        for i in np.flatnonzero(unique_diffs >= gap_factor * interval)
    )
    expected = max(1.0, (float(times[-1]) - float(times[0])) / interval + 1.0)
    coverage = min(1.0, float(np.unique(times).size) / expected)
    return SamplingReport(
        count=int(times.size),
        start=float(times[0]),
        end=float(times[-1]),
        median_interval=interval,
        jitter_mad=jitter,
        coverage=coverage,
        duplicate_count=duplicate_count,
        gaps=gaps,
    )


def _telemetry(
    rows: Iterable[dict[str, Any]], timestamp: str, value: str, source: str
) -> Telemetry:
    pairs: list[tuple[float, float]] = []
    for line, row in enumerate(rows, 2):
        try:
            pairs.append((parse_timestamp(row[timestamp]), float(row[value])))
        except KeyError as error:
            raise ValueError(f"missing column {error.args[0]!r}") from error
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid value on row {line}: {error}") from error
    if len(pairs) < 3:
        raise ValueError("input must contain at least three observations")
    pairs.sort()
    times = np.asarray([pair[0] for pair in pairs], dtype=float)
    values = np.asarray([pair[1] for pair in pairs], dtype=float)
    return Telemetry(times, values, name=value, source=source, sampling=sampling_report(times))


def read_csv(
    path_or_text: str | Path,
    *,
    timestamp: str = "timestamp",
    value: str = "value",
) -> Telemetry:
    """Read timestamped telemetry from a CSV path or CSV text."""
    path = Path(path_or_text)
    if "\n" not in str(path_or_text) and path.exists():
        text = path.read_text(encoding="utf-8-sig")
        source = str(path)
    else:
        text = str(path_or_text)
        source = "<memory>"
    return _telemetry(csv.DictReader(io.StringIO(text)), timestamp, value, source)


def read_jsonl(
    path_or_text: str | Path,
    *,
    timestamp: str = "timestamp",
    value: str = "value",
) -> Telemetry:
    """Read one JSON object per line from a path or string."""
    path = Path(path_or_text)
    if "\n" not in str(path_or_text) and path.exists():
        text = path.read_text(encoding="utf-8")
        source = str(path)
    else:
        text = str(path_or_text)
        source = "<memory>"
    rows: list[dict[str, Any]] = []
    for line, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line}") from error
        if not isinstance(item, dict):
            raise ValueError(f"JSONL line {line} is not an object")
        rows.append(item)
    return _telemetry(rows, timestamp, value, source)


def read_telemetry(
    path: str | Path,
    *,
    timestamp: str = "timestamp",
    value: str = "value",
) -> Telemetry:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return read_csv(path, timestamp=timestamp, value=value)
    if suffix in {".jsonl", ".ndjson"}:
        return read_jsonl(path, timestamp=timestamp, value=value)
    raise ValueError("unsupported format; use .csv, .jsonl, or .ndjson")
