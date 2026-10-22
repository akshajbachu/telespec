"""Command-line interface for telespec."""

from __future__ import annotations

import csv
import json
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .analysis import analyze
from .findings import write_report
from .io import read_telemetry
from .models import AnalysisResult
from .spectral import human_period
from .validation import Injection, generate

app = typer.Typer(no_args_is_help=True, help="Explainable spectral analysis for telemetry.")
console = Console()


def _run(path: Path, column: str, timestamp: str, bootstraps: int) -> AnalysisResult:
    telemetry = read_telemetry(path, timestamp=timestamp, value=column)
    return analyze(telemetry, bootstrap_iterations=bootstraps)


@app.command()
def analyze_file(
    path: Path,
    column: Annotated[str, typer.Option()] = "value",
    timestamp: Annotated[str, typer.Option()] = "timestamp",
    bootstraps: Annotated[int, typer.Option(min=10)] = 100,
) -> None:
    """Analyze a CSV/JSONL file and print findings."""
    result = _run(path, column, timestamp, bootstraps)
    sampling = result.telemetry.sampling
    if sampling:
        console.print(
            f"[bold]Sampling[/bold] {sampling.count} points, "
            f"{sampling.coverage:.1%} coverage, jitter MAD {sampling.jitter_mad:.3g}s"
        )
    table = Table("Kind", "Interval", "Confidence", "Summary")
    for finding in result.findings:
        table.add_row(
            finding.kind,
            f"{finding.start:.0f}-{finding.end:.0f}",
            f"{finding.confidence:.1%}",
            finding.summary,
        )
    console.print(table)


@app.command()
def periods(
    path: Path,
    column: Annotated[str, typer.Option()] = "value",
    timestamp: Annotated[str, typer.Option()] = "timestamp",
    bootstraps: Annotated[int, typer.Option(min=10)] = 100,
) -> None:
    """List statistically significant periods."""
    result = _run(path, column, timestamp, bootstraps)
    table = Table("Period", "Frequency", "Power", "FAP", "Harmonics")
    for peak in result.peaks:
        table.add_row(
            human_period(peak.period),
            f"{peak.frequency:.6g}",
            f"{peak.power:.4f}",
            f"{peak.false_alarm_probability:.4g}",
            ", ".join(f"{value:.4g}" for value in peak.harmonics) or "-",
        )
    console.print(table)


@app.command()
def anomalies(
    path: Path,
    column: Annotated[str, typer.Option()] = "value",
    timestamp: Annotated[str, typer.Option()] = "timestamp",
    train_frac: Annotated[float, typer.Option(min=0.05, max=0.9)] = 0.3,
    format: Annotated[str, typer.Option()] = "json",
) -> None:
    """Print non-periodicity findings as JSON or Markdown."""
    telemetry = read_telemetry(path, timestamp=timestamp, value=column)
    result = analyze(telemetry, train_fraction=train_frac)
    selected = [finding.to_dict() for finding in result.findings if finding.kind != "periodicity"]
    if format == "json":
        console.print_json(json.dumps(selected))
    else:
        for finding in selected:
            console.print(f"- {finding['summary']}")


@app.command()
def report(
    path: Path,
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("telespec-report.md"),
    column: Annotated[str, typer.Option()] = "value",
    timestamp: Annotated[str, typer.Option()] = "timestamp",
) -> None:
    """Write a reproducible Markdown or JSON report."""
    result = _run(path, column, timestamp, 100)
    write_report(result.findings, output, title=f"telespec report: {path.name}")
    console.print(f"Wrote {output}")


@app.command()
def demo(output: Annotated[Path, typer.Option("-o", "--output")] = Path("demo.csv")) -> None:
    """Generate and analyze a telemetry recording with known effects."""
    synthetic = generate(
        injections=(
            Injection("periodicity", 0.45, 0.75, 8.0, 900.0),
            Injection("level_shift", 0.72, 0.92, 12.0),
        )
    )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "latency"])
        writer.writerows(
            zip(synthetic.telemetry.timestamps, synthetic.telemetry.values, strict=True)
        )
    console.print(f"Wrote {output}; run: telespec analyze-file {output} --column latency")


@app.command()
def view(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Launch the private local dashboard."""
    try:
        import uvicorn
    except ImportError as error:
        raise typer.BadParameter("Install telespec[ui] to use the dashboard.") from error
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    uvicorn.run("telespec.server.app:app", host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
