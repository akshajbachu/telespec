"""Regenerate synthetic and real-data validation artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from datasets import DEFAULT_SERIES, fetch_nab

from telespec.analysis import analyze
from telespec.detect import ewma_baseline, rolling_robust_z, sigma_baseline
from telespec.io import parse_timestamp, read_csv, sampling_report
from telespec.models import Telemetry
from telespec.spectral import bootstrap_threshold, generalized_lomb_scargle
from telespec.validation import Injection, generate

RESULTS = Path("benchmarks/results")


def frequency_recovery(trials: int) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    truth = 1.0 / 1800.0
    for missing in (0.0, 0.2, 0.4, 0.6):
        errors = []
        fft_errors = []
        for seed in range(trials):
            sample = generate(
                points=240,
                interval=60,
                daily_amplitude=0,
                noise=1,
                missing=missing,
                jitter=0.2,
                injections=(Injection("periodicity", 0, 1, 5, 1800),),
                random_state=seed,
            ).telemetry
            grid = np.linspace(truth * 0.3, truth * 2.0, 160)
            gls = generalized_lomb_scargle(sample.timestamps, sample.values, grid)
            errors.append(abs(grid[np.argmax(gls.power)] - truth) / truth)
            uniform_t = np.arange(sample.timestamps[0], sample.timestamps[-1], 60.0)
            interpolated = np.interp(uniform_t, sample.timestamps, sample.values)
            fft_f = np.fft.rfftfreq(interpolated.size, 60.0)[1:]
            fft_p = np.abs(np.fft.rfft(interpolated - interpolated.mean()))[1:] ** 2
            fft_errors.append(abs(fft_f[np.argmax(fft_p)] - truth) / truth)
        records.append(
            {
                "missing": missing,
                "gls_median_relative_error": float(np.median(errors)),
                "interpolate_fft_median_relative_error": float(np.median(fft_errors)),
            }
        )
    return records


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def fap_calibration(trials: int) -> list[dict[str, float]]:
    """Measure empirical family-wise false alarms on pure noise."""
    records: list[dict[str, float]] = []
    for nominal in (0.01, 0.05, 0.1):
        alarms = 0
        for seed in range(trials):
            sample = generate(
                points=120,
                interval=60,
                daily_amplitude=0,
                noise=1,
                random_state=seed + 1000,
            ).telemetry
            frequencies = np.linspace(1 / 5000, 1 / 180, 80)
            periodogram = generalized_lomb_scargle(sample.timestamps, sample.values, frequencies)
            threshold, _ = bootstrap_threshold(
                sample.timestamps,
                sample.values,
                frequencies,
                false_alarm_probability=nominal,
                iterations=max(20, trials * 2),
                random_state=seed,
            )
            alarms += bool(np.max(periodogram.power) >= threshold)
        records.append(
            {
                "nominal_false_alarm_probability": nominal,
                "empirical_false_alarm_rate": alarms / trials,
                "trials": float(trials),
            }
        )
    return records


def synthetic_detection(trials: int) -> dict[str, dict[str, dict[str, float]]]:
    """Score each effect independently against generic point-detector baselines."""
    configurations = {
        "level_shift": Injection("level_shift", 0.65, 0.9, 7),
        "variance_burst": Injection("variance_burst", 0.65, 0.9, 6),
        "spikes": Injection("spikes", 0.65, 0.9, 12),
        "periodicity": Injection("periodicity", 0.65, 0.9, 8, 600),
    }
    output: dict[str, dict[str, dict[str, float]]] = {}
    for kind, injection in configurations.items():
        counts = {
            "telespec": [0, 0, 0],
            "sigma": [0, 0, 0],
            "ewma": [0, 0, 0],
            "rolling_robust_z": [0, 0, 0],
        }
        for seed in range(trials):
            sample = generate(
                points=240,
                interval=60,
                daily_amplitude=0,
                injections=(injection,),
                random_state=seed,
            ).telemetry
            truth_start = sample.timestamps[round(sample.timestamps.size * 0.65)]
            truth_end = sample.timestamps[
                min(sample.timestamps.size - 1, round(sample.timestamps.size * 0.9))
            ]
            result = analyze(sample, bootstrap_iterations=20)
            telespec_events = [
                (finding.start + finding.end) / 2
                for finding in result.findings
                if finding.kind != "periodicity" or kind == "periodicity"
            ]
            detectors = {
                "telespec": np.asarray(telespec_events),
                "sigma": sample.timestamps[sigma_baseline(sample.values) > 3],
                "ewma": sample.timestamps[ewma_baseline(sample.values) > 3],
                "rolling_robust_z": sample.timestamps[rolling_robust_z(sample.values) > 3],
            }
            for name, events in detectors.items():
                inside = np.asarray((events >= truth_start) & (events <= truth_end), dtype=bool)
                counts[name][0] += bool(np.any(inside))
                counts[name][1] += bool(np.any(~inside))
                counts[name][2] += not np.any(inside)
        output[kind] = {name: _metrics(tp, fp, fn) for name, (tp, fp, fn) in counts.items()}
    return output


def _event_score(
    events: np.ndarray,
    windows: list[tuple[float, float]],
) -> dict[str, float]:
    matches = [any(start <= event <= end for event in events) for start, end in windows]
    tp = sum(matches)
    fn = len(windows) - tp
    fp = sum(not any(start <= event <= end for start, end in windows) for event in events)
    return _metrics(tp, fp, fn)


def _group_events(timestamps: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indexes = np.flatnonzero(mask)
    if not indexes.size:
        return np.empty(0)
    groups = np.split(indexes, np.flatnonzero(np.diff(indexes) > 1) + 1)
    return np.asarray([timestamps[group[len(group) // 2]] for group in groups])


def nab_evaluation(limit: int) -> list[dict[str, object]]:
    records = []
    for labeled in fetch_nab(series=DEFAULT_SERIES[:limit]):
        telemetry = read_csv(labeled.path)
        stride = max(1, int(np.ceil(telemetry.values.size / 720)))
        telemetry = Telemetry(
            telemetry.timestamps[::stride],
            telemetry.values[::stride],
            telemetry.name,
            telemetry.unit,
            telemetry.source,
            sampling_report(telemetry.timestamps[::stride]),
        )
        result = analyze(telemetry, bootstrap_iterations=20)
        windows = [(parse_timestamp(start), parse_timestamp(end)) for start, end in labeled.windows]
        events = np.asarray(
            [
                (finding.start + finding.end) / 2
                for finding in result.findings
                if finding.kind != "periodicity"
            ]
        )
        baseline_events = {
            "sigma": _group_events(telemetry.timestamps, sigma_baseline(telemetry.values) > 3),
            "ewma": _group_events(telemetry.timestamps, ewma_baseline(telemetry.values) > 3),
            "rolling_robust_z": _group_events(
                telemetry.timestamps, rolling_robust_z(telemetry.values) > 3
            ),
        }
        records.append(
            {
                "series": str(labeled.path),
                "labeled_windows": len(labeled.windows),
                "downsample_stride": stride,
                "telespec": _event_score(events, windows),
                "baselines": {
                    name: _event_score(candidate, windows)
                    for name, candidate in baseline_events.items()
                },
                "sha256": labeled.sha256,
            }
        )
    return records


def write_svg(records: list[dict[str, float]]) -> None:
    width, height = 640, 340
    bars = []
    for index, record in enumerate(records):
        x = 80 + index * 130
        for offset, key, color in (
            (0, "gls_median_relative_error", "#2dd4bf"),
            (38, "interpolate_fft_median_relative_error", "#fb7185"),
        ):
            value = min(1.0, record[key])
            bar_height = value * 220
            bars.append(
                f'<rect x="{x + offset}" y="{280 - bar_height:.1f}" width="30" '
                f'height="{bar_height:.1f}" fill="{color}"/>'
            )
        bars.append(
            f'<text x="{x + 25}" y="305" text-anchor="middle" fill="#334155">'
            f"{record['missing']:.0%}</text>"
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        '<text x="24" y="30" font-family="sans-serif" font-size="18" fill="#0f172a">'
        "Frequency recovery under missing samples</text>"
        + "".join(bars)
        + '<text x="320" y="330" text-anchor="middle" font-family="sans-serif" '
        'font-size="13" fill="#334155">Missing observations</text></svg>'
    )
    (RESULTS / "frequency_recovery.svg").write_text(svg, encoding="utf-8")


def update_readme(result: dict[str, object]) -> None:
    readme = Path("README.md")
    if not readme.exists():
        return
    recovery = result["frequency_recovery"]
    assert isinstance(recovery, list)
    hardest = recovery[-1]
    synthetic = result["synthetic_detection"]
    assert isinstance(synthetic, dict)
    lines = [
        "<!-- BENCHMARK_RESULTS_START -->",
        f"Latest `{result['mode']}` reproducible run:",
        "",
        (
            f"- At {hardest['missing']:.0%} missing observations, median frequency error was "
            f"{hardest['gls_median_relative_error']:.2%} for GLS versus "
            f"{hardest['interpolate_fft_median_relative_error']:.2%} for interpolate+FFT."
        ),
    ]
    for kind, detectors in synthetic.items():
        lines.append(
            f"- Synthetic {kind.replace('_', ' ')} F1: "
            f"{detectors['telespec']['f1']:.3f} (telespec), "
            f"{detectors['sigma']['f1']:.3f} (3-sigma), "
            f"{detectors['ewma']['f1']:.3f} (EWMA)."
        )
    nab = result["nab"]
    assert isinstance(nab, list)
    if nab:
        telespec_f1 = float(np.mean([record["telespec"]["f1"] for record in nab]))
        lines.append(f"- NAB macro F1 across {len(nab)} selected series: {telespec_f1:.3f}.")
    lines.append("<!-- BENCHMARK_RESULTS_END -->")
    text = readme.read_text(encoding="utf-8")
    start = text.index("<!-- BENCHMARK_RESULTS_START -->")
    end = text.index("<!-- BENCHMARK_RESULTS_END -->") + len("<!-- BENCHMARK_RESULTS_END -->")
    readme.write_text(text[:start] + "\n".join(lines) + text[end:], encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--with-nab", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    trials = 3 if args.quick else 20
    recovery = frequency_recovery(trials)
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "quick" if args.quick else "full",
        "frequency_recovery": recovery,
        "fap_calibration": fap_calibration(trials),
        "synthetic_detection": synthetic_detection(trials),
        "nab": nab_evaluation(2 if args.quick else 5) if args.with_nab else [],
    }
    (RESULTS / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_svg(recovery)
    update_readme(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
