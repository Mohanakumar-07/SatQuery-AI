"""On-demand downloadable analysis reports."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from app.api.v1.deps import AppSettings, DbSession, require_analysis
from app.core.storage import get_store
from app.services.report_service import get_report_service

router = APIRouter(prefix="/analyses/{analysis_id}", tags=["reports"])


@router.get("/report", summary="Download a JSON, HTML or PDF analysis report")
def report(
    analysis_id: str,
    session: DbSession,
    settings: AppSettings,
    format: Literal["json", "html", "pdf"] = Query(default="html"),
    download: bool = Query(default=True),
) -> FileResponse:
    analysis = require_analysis(session, analysis_id)
    artifact = get_report_service(settings).ensure_report(session, analysis, format=format)
    session.commit()
    path = get_store(settings).from_relative(artifact.relative_path)
    disposition = "attachment" if download else "inline"
    return FileResponse(
        path=path,
        media_type=artifact.media_type,
        filename=None,
        headers={
            "Content-Disposition": f'{disposition}; filename="{artifact.name}"',
            "X-Content-Type-Options": "nosniff",
        },
    )
