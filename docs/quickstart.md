# Quickstart

```bash
uv sync --extra dev
uv run telespec demo -o demo.csv
uv run telespec analyze-file demo.csv --column latency
uv run telespec periods demo.csv --column latency
uv run telespec report demo.csv --column latency -o report.md
uv run telespec view
```

CSV and JSONL files require one timestamp column and one numeric metric column.
Timestamps may be ISO-8601, epoch seconds, or epoch milliseconds.
