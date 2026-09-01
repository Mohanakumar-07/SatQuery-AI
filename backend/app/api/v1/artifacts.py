"""Analysis artifact listing and containment-checked file delivery."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from app.api.v1.deps import AppSettings, DbSession, require_analysis
from app.core.errors import ArtifactNotFound
from app.core.ids import safe_filename
from app.core.storage import get_store, is_inline_safe
from app.db.repo import get_artifact, list_artifacts
from app.schemas.analyses import ArtifactLink, ArtifactListResponse
from app.services.result_service import get_result_service

router = APIRouter(prefix="/analyses/{analysis_id}/artifacts", tags=["artifacts"])


@router.get("", response_model=ArtifactListResponse, summary="List analysis artifacts")
def artifacts(
    analysis_id: str,
    session: DbSession,
    settings: AppSettings,
    kind: str | None = Query(default=None, max_length=32),
) -> ArtifactListResponse:
    require_analysis(session, analysis_id)
    rows = list_artifacts(session, analysis_id, kind=kind)
    by_id = {item.artifact_id: item for item in get_result_service(settings).artifact_links(session, analysis_id)}
    items = [by_id[row.id] for row in rows if row.id in by_id]
    return ArtifactListResponse(items=items, total=len(items))


@router.get("/{artifact_id}", summary="Download or display an analysis artifact")
def artifact_file(
    analysis_id: str,
    artifact_id: str,
    session: DbSession,
    settings: AppSettings,
    inline: bool | None = Query(default=None),
) -> FileResponse:
    require_analysis(session, analysis_id)
    artifact = get_artifact(session, analysis_id, artifact_id)
    if artifact is None:
        raise ArtifactNotFound(detail={"analysis_id": analysis_id, "artifact_id": artifact_id})
    path = get_store(settings).from_relative(artifact.relative_path)
    if not path.is_file():
        raise ArtifactNotFound(
            "The artifact record exists, but its stored file is missing.",
            detail={"analysis_id": analysis_id, "artifact_id": artifact_id},
        )
    render_inline = is_inline_safe(artifact.media_type) if inline is None else inline
    disposition = "inline" if render_inline else "attachment"
    filename = safe_filename(artifact.name, fallback="artifact.bin")
    headers = {
        "Content-Disposition": f'{disposition}; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=3600",
    }
    if artifact.sha256:
        headers["ETag"] = f'"{artifact.sha256}"'
    return FileResponse(path=path, media_type=artifact.media_type, filename=None, headers=headers)
