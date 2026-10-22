from pathlib import Path

from typer.testing import CliRunner

import telespec.cli as cli
from telespec.analysis import analyze
from telespec.validation import Injection, generate


def _result():
    telemetry = generate(
        points=80,
        interval=60,
        daily_amplitude=0,
        noise=0.1,
        injections=(Injection("periodicity", 0, 1, 5, 600),),
    ).telemetry
    return analyze(telemetry, bootstrap_iterations=10)


def test_analysis_periods_anomalies_and_report(monkeypatch, tmp_path):
    runner = CliRunner()
    result = _result()
    source = tmp_path / "metric.csv"
    source.write_text("timestamp,value\n0,1\n1,2\n2,3\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_run", lambda *args, **kwargs: result)
    monkeypatch.setattr(cli, "analyze", lambda *args, **kwargs: result)
    monkeypatch.setattr(cli, "read_telemetry", lambda *args, **kwargs: result.telemetry)

    analyzed = runner.invoke(cli.app, ["analyze-file", str(source), "--bootstraps", "10"])
    assert analyzed.exit_code == 0
    assert "Sampling" in analyzed.stdout

    periods = runner.invoke(cli.app, ["periods", str(source), "--bootstraps", "10"])
    assert periods.exit_code == 0
    assert "Frequency" in periods.stdout

    anomalies = runner.invoke(cli.app, ["anomalies", str(source), "--format", "json"])
    assert anomalies.exit_code == 0

    markdown = runner.invoke(cli.app, ["anomalies", str(source), "--format", "markdown"])
    assert markdown.exit_code == 0

    output = tmp_path / "report.md"
    report = runner.invoke(cli.app, ["report", str(source), "-o", str(output)])
    assert report.exit_code == 0
    assert output.exists()


def test_demo_and_view(monkeypatch, tmp_path):
    runner = CliRunner()
    demo = tmp_path / "demo.csv"
    generated = runner.invoke(cli.app, ["demo", "-o", str(demo)])
    assert generated.exit_code == 0
    assert demo.read_text(encoding="utf-8").startswith("timestamp,latency")

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        "uvicorn.run",
        lambda _app, host, port: calls.append((host, port)),
    )
    monkeypatch.setattr(cli.webbrowser, "open", lambda _url: True)
    viewed = runner.invoke(cli.app, ["view", "--port", "9001", "--open"])
    assert viewed.exit_code == 0
    assert calls == [("127.0.0.1", 9001)]


def test_run_reads_file(tmp_path: Path):
    source = tmp_path / "metric.csv"
    source.write_text(
        "timestamp,value\n" + "\n".join(f"{i * 60},{i % 4}" for i in range(40)),
        encoding="utf-8",
    )
    assert cli._run(source, "value", "timestamp", 10).telemetry.values.size == 40
