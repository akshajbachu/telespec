"""Reproducible downloader for selected Numenta Anomaly Benchmark series."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

NAB_RAW = "https://raw.githubusercontent.com/numenta/NAB/master"
DEFAULT_SERIES = (
    "realAWSCloudwatch/ec2_network_in_257a54.csv",
    "realAWSCloudwatch/ec2_network_in_5abac7.csv",
    "realAWSCloudwatch/elb_request_count_8c0756.csv",
    "realKnownCause/ec2_request_latency_system_failure.csv",
    "realTraffic/speed_6005.csv",
)


@dataclass(frozen=True)
class LabeledSeries:
    path: Path
    windows: tuple[tuple[str, str], ...]
    sha256: str


def _download(url: str, path: Path) -> str:
    data = urlopen(url, timeout=30).read()
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def fetch_nab(
    cache: str | Path = "benchmarks/data/nab",
    *,
    series: tuple[str, ...] = DEFAULT_SERIES,
) -> list[LabeledSeries]:
    root = Path(cache)
    labels_path = root / "combined_windows.json"
    if not labels_path.exists():
        _download(f"{NAB_RAW}/labels/combined_windows.json", labels_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    loaded: list[LabeledSeries] = []
    for relative in series:
        target = root / relative
        if not target.exists():
            digest = _download(f"{NAB_RAW}/data/{relative}", target)
        else:
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        windows = tuple((str(start), str(end)) for start, end in labels.get(relative, []))
        loaded.append(LabeledSeries(target, windows, digest))
    return loaded
