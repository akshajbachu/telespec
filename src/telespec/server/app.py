"""FastAPI application for the local telespec dashboard."""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from ..analysis import analyze
from ..findings import findings_markdown
from ..io import read_csv, read_jsonl
from ..validation import Injection, generate

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_STORED_ANALYSES = 20
ANALYSES: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _remember(result: dict[str, Any]) -> None:
    ANALYSES[result["id"]] = result
    ANALYSES.move_to_end(result["id"])
    while len(ANALYSES) > MAX_STORED_ANALYSES:
        ANALYSES.popitem(last=False)


def _demo_result() -> dict[str, Any]:
    synthetic = generate(
        points=360,
        injections=(
            Injection("periodicity", 0.4, 0.75, 8.0, 900.0),
            Injection("level_shift", 0.72, 0.94, 12.0),
        ),
    )
    return analyze(synthetic.telemetry, bootstrap_iterations=30).to_dict()


def create_app(static_dir: Path | None = None) -> FastAPI:
    application = FastAPI(
        title="telespec local dashboard",
        version="0.1.0",
        description="Local-only spectral telemetry analysis API.",
    )

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "telespec", "local_only": True, "version": "0.1.0"}

    @application.post("/api/analyze")
    async def analyze_upload(
        file: Annotated[UploadFile, File()],
        timestamp: Annotated[str, Form()] = "timestamp",
        column: Annotated[str, Form()] = "value",
        train_fraction: Annotated[float, Form()] = 0.3,
        false_alarm_probability: Annotated[float, Form()] = 0.01,
        bootstraps: Annotated[int, Form()] = 100,
    ) -> dict[str, Any]:
        if not 0.05 <= train_fraction <= 0.9:
            raise HTTPException(422, "train_fraction must be between 0.05 and 0.9")
        if not 0.0001 <= false_alarm_probability <= 0.2:
            raise HTTPException(422, "false_alarm_probability must be between 0.0001 and 0.2")
        if not 10 <= bootstraps <= 2000:
            raise HTTPException(422, "bootstraps must be between 10 and 2000")
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "upload exceeds the 20 MB local limit")
        try:
            text = content.decode("utf-8-sig")
            suffix = Path(file.filename or "").suffix.lower()
            telemetry = (
                read_jsonl(text, timestamp=timestamp, value=column)
                if suffix in {".jsonl", ".ndjson"}
                else read_csv(text, timestamp=timestamp, value=column)
            )
            result = analyze(
                telemetry,
                train_fraction=train_fraction,
                false_alarm_probability=false_alarm_probability,
                bootstrap_iterations=bootstraps,
            ).to_dict()
        except (UnicodeDecodeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
        result["id"] = uuid.uuid4().hex
        result["filename"] = Path(file.filename or "telemetry").name
        _remember(result)
        return result

    @application.post("/api/demo")
    def demo() -> dict[str, Any]:
        result = _demo_result()
        result["id"] = uuid.uuid4().hex
        result["filename"] = "telespec-demo.csv"
        _remember(result)
        return result

    @application.get("/api/export/{analysis_id}")
    def export(
        analysis_id: str,
        format: Literal["json", "markdown"] = "json",
    ) -> Response:
        result = ANALYSES.get(analysis_id)
        if result is None:
            raise HTTPException(404, "analysis not found in this local session")
        if format == "markdown":
            from ..models import Finding

            findings = [Finding(**item) for item in result["findings"]]
            text = findings_markdown(findings, title=f"telespec report: {result['filename']}")
            return Response(
                text,
                media_type="text/markdown",
                headers={"Content-Disposition": 'attachment; filename="telespec-report.md"'},
            )
        return Response(
            json.dumps(result, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="telespec-analysis.json"'},
        )

    assets = static_dir if static_dir is not None else STATIC_DIR
    if (assets / "index.html").is_file():
        application.mount("/", StaticFiles(directory=assets, html=True), name="dashboard")
    else:

        @application.get("/")
        def root() -> dict[str, str]:
            return {"service": "telespec", "message": "Run npm run build in frontend."}

    return application


app = create_app()
