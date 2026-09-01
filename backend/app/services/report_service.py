"""Downloadable analysis reports backed by the stored result contract."""

from __future__ import annotations

import html
import io
import json
import textwrap
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import Settings, get_settings
from app.core.errors import ReportFormatUnavailable
from app.core.storage import ArtifactStore, get_store
from app.db.models import Analysis, Artifact
from app.db.repo import get_artifact_by_name
from app.services.result_service import ResultService, get_result_service

ReportFormat = Literal["json", "html", "pdf"]


@dataclass(frozen=True)
class RenderedReport:
    name: str
    media_type: str
    content: bytes


class ReportService:
    """Renders immutable reports from the final public result payload."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: ArtifactStore | None = None,
        results: ResultService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store(self.settings)
        self.results = results or get_result_service(self.settings)

    def available_formats(self) -> list[str]:
        formats = ["json", "html"]
        try:
            import reportlab  # noqa: F401
        except ImportError:
            return formats
        return [*formats, "pdf"]

    def ensure_report(self, session, analysis: Analysis, *, format: ReportFormat = "html") -> Artifact:
        """Return a stable stored report, generating it once when necessary."""
        name = f"satquery-{analysis.id}.{format}"
        existing = get_artifact_by_name(session, analysis.id, name)
        if existing is not None and self.store.from_relative(existing.relative_path).is_file():
            return existing

        payload = self.results.build_result(session, analysis).model_dump(mode="json")
        rendered = self.render(payload, format=format, name=name)
        authoritative = bool((payload.get("pipeline") or {}).get("authoritative"))
        self.results.register_artifact_files(
            session,
            analysis,
            [
                {
                    "name": rendered.name,
                    "data": rendered.content,
                    "kind": "report",
                    "source": "report_service",
                    "media_type": rendered.media_type,
                    "synthetic": not authoritative,
                    "description": f"SatQuery {format.upper()} analysis report",
                }
            ],
        )
        artifact = get_artifact_by_name(session, analysis.id, name)
        if artifact is None:  # defensive: registration is expected to flush the row
            raise RuntimeError("The report was rendered but could not be registered.")
        return artifact

    def render(self, payload: dict[str, Any], *, format: ReportFormat, name: str) -> RenderedReport:
        if format == "json":
            content = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
            return RenderedReport(name=name, media_type="application/json", content=content)
        if format == "html":
            return RenderedReport(
                name=name,
                media_type="text/html; charset=utf-8",
                content=self._html(payload).encode("utf-8"),
            )
        if format == "pdf":
            return RenderedReport(name=name, media_type="application/pdf", content=self._pdf(payload))
        raise ReportFormatUnavailable(
            f"Report format '{format}' is not supported.",
            detail={"available_formats": self.available_formats()},
        )

    def _html(self, payload: dict[str, Any]) -> str:
        title = f"SatQuery analysis {payload.get('analysis_id', '')}"
        warnings = payload.get("warnings") or []
        artifacts = payload.get("artifacts") or []
        warning_items = "".join(
            f"<li><strong>{html.escape(str(item.get('code', 'WARNING')))}</strong>: "
            f"{html.escape(str(item.get('message', '')))}</li>"
            for item in warnings
            if isinstance(item, dict)
        ) or "<li>None</li>"
        artifact_items = "".join(
            f"<li>{html.escape(str(item.get('name', 'artifact')))} "
            f"({html.escape(str(item.get('kind', 'other')))})</li>"
            for item in artifacts
            if isinstance(item, dict)
        ) or "<li>None</li>"
        evidence_json = html.escape(json.dumps(payload.get("evidence"), indent=2, ensure_ascii=False, default=str))
        confidence_json = html.escape(
            json.dumps(payload.get("confidence"), indent=2, ensure_ascii=False, default=str)
        )
        trace = "".join(f"<li>{html.escape(str(step))}</li>" for step in payload.get("execution_trace") or [])
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
    body {{ max-width: 980px; margin: 2rem auto; padding: 0 1.5rem; color: #111; }}
    header {{ border-bottom: 2px solid #111; margin-bottom: 1.5rem; }}
    h1, h2 {{ line-height: 1.2; }}
    .answer {{ padding: 1rem; background: #f2f2f2; border-left: 4px solid #222; }}
    dl {{ display: grid; grid-template-columns: 12rem 1fr; gap: .4rem 1rem; }}
    dt {{ font-weight: 700; }} dd {{ margin: 0; }}
    pre {{ overflow-wrap: anywhere; white-space: pre-wrap; background: #f7f7f7; padding: 1rem; }}
    footer {{ border-top: 1px solid #999; margin-top: 2rem; padding-top: 1rem; color: #555; }}
    @media print {{ body {{ margin: 0; max-width: none; }} }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1><p>Evidence-backed spatial analysis report</p></header>
  <dl>
    <dt>Status</dt><dd>{html.escape(str(payload.get('status', 'unknown')))}</dd>
    <dt>Task</dt><dd>{html.escape(str(payload.get('task') or 'unresolved'))}</dd>
    <dt>Question</dt><dd>{html.escape(str(payload.get('question') or ''))}</dd>
    <dt>Created</dt><dd>{html.escape(str(payload.get('created_at') or ''))}</dd>
    <dt>Finished</dt><dd>{html.escape(str(payload.get('finished_at') or ''))}</dd>
  </dl>
  <h2>Answer</h2><p class="answer">{html.escape(str(payload.get('answer') or 'No answer produced.'))}</p>
  <h2>Evidence</h2><pre>{evidence_json}</pre>
  <h2>Confidence policy outcome</h2><pre>{confidence_json}</pre>
  <h2>Warnings</h2><ul>{warning_items}</ul>
  <h2>Artifacts</h2><ul>{artifact_items}</ul>
  <h2>Execution trace</h2><ol>{trace}</ol>
  <footer>{html.escape(str(payload.get('disclaimer') or 'Generated by SatQuery AI.'))}</footer>
</body>
</html>"""

    def _pdf(self, payload: dict[str, Any]) -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
        except ImportError as exc:
            raise ReportFormatUnavailable(
                "PDF reports require the optional 'reportlab' package.",
                detail={"available_formats": self.available_formats(), "install": "pip install reportlab"},
            ) from exc

        buffer = io.BytesIO()
        document = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        margin = 48
        y = height - margin

        def line(text: str = "", *, size: int = 9, leading: int = 12) -> None:
            nonlocal y
            document.setFont("Helvetica", size)
            for wrapped in textwrap.wrap(str(text), width=100, replace_whitespace=False) or [""]:
                if y < margin:
                    document.showPage()
                    y = height - margin
                    document.setFont("Helvetica", size)
                document.drawString(margin, y, wrapped)
                y -= leading

        document.setTitle(f"SatQuery analysis {payload.get('analysis_id', '')}")
        line("SatQuery AI analysis report", size=16, leading=22)
        line(f"Analysis: {payload.get('analysis_id', '')}", size=10)
        line(f"Status: {payload.get('status', '')}    Task: {payload.get('task') or 'unresolved'}", size=10)
        line(f"Question: {payload.get('question') or ''}", size=10)
        y -= 8
        line("Answer", size=13, leading=18)
        line(str(payload.get("answer") or "No answer produced."), size=10, leading=14)
        y -= 8
        line("Evidence and confidence", size=13, leading=18)
        summary = {
            "evidence": payload.get("evidence"),
            "confidence": payload.get("confidence"),
            "warnings": payload.get("warnings"),
            "execution_trace": payload.get("execution_trace"),
            "versions": payload.get("versions"),
            "disclaimer": payload.get("disclaimer"),
        }
        for raw_line in json.dumps(summary, indent=2, ensure_ascii=True, default=str).splitlines():
            line(raw_line, size=7, leading=9)
        document.save()
        return buffer.getvalue()


_service: ReportService | None = None


def get_report_service(settings: Settings | None = None) -> ReportService:
    global _service
    if _service is None:
        _service = ReportService(settings=settings)
    return _service


def reset_report_service() -> None:
    global _service
    _service = None
