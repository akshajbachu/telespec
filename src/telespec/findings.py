"""Finding serialization and human-readable reports."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .models import Finding


def findings_json(findings: Iterable[Finding]) -> str:
    return json.dumps([finding.to_dict() for finding in findings], indent=2)


def findings_markdown(findings: Iterable[Finding], *, title: str = "telespec report") -> str:
    items = list(findings)
    lines = [f"# {title}", "", f"{len(items)} interpretable findings.", ""]
    for item in items:
        lines.extend(
            [
                f"## {item.kind.replace('_', ' ').title()}",
                "",
                item.summary,
                "",
                f"- Interval: `{item.start:.3f}` to `{item.end:.3f}`",
                f"- Magnitude: `{item.magnitude:.4g}`",
                f"- Confidence: `{item.confidence:.1%}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_report(
    findings: Iterable[Finding],
    path: str | Path,
    *,
    title: str = "telespec report",
) -> Path:
    output = Path(path)
    text = (
        findings_json(findings)
        if output.suffix.lower() == ".json"
        else findings_markdown(findings, title=title)
    )
    output.write_text(text, encoding="utf-8")
    return output
