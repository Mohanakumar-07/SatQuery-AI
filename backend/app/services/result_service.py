"""Result service: save answer, evidence, artifacts and execution trace (section 7.1).

Called by the worker when a pipeline finishes, and by ``GET /analyses/{id}/result`` to
read the stored payload back. It is the single place where the section 7.5 response is
assembled, so both paths are guaranteed to agree.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.errors import BadRequest, Conflict
from app.core.ids import new_id, safe_filename
from app.core.logging import get_logger
from app.core.storage import ArtifactStore, get_store, media_type_for
from app.db.models import Analysis, Artifact, Upload
from app.db.repo import (
    add_warnings,
    append_trace,
    complete,
    get_analysis,
    list_artifacts,
    register_artifact,
)
from app.geospatial.overlap import relative_location
from app.schemas.analyses import AnalysisResult, ArtifactLink, InputInterpretation, PipelineInfo
from app.schemas.common import (
    AnalysisStatus,
    AnswerType,
    ConfidenceDecision,
    FileRole,
    InputType,
    ModelRef,
    Modality,
    Stage,
    VersionBundle,
    Warning,
    WarningLevel,
)
from app.schemas.evidence import Evidence
from app.services.confidence_service import ConfidenceService, get_confidence_service
from app.services.evidence_service import get_evidence_service
from app.services.validation_service import ValidationResponse

logger = get_logger("services.result")

STUB_DISCLAIMER = (
    "Stub pipeline output. The response shape is real; the answer, evidence and "
    "confidence are placeholders produced with no model inference. Attach the "
    "pipeline (SATQUERY_PIPELINE_MODE=python) before using any result."
)


class ResultService:
    """Persists analysis outcomes and rebuilds the public result payload."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: ArtifactStore | None = None,
        confidence: ConfidenceService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store(self.settings)
        self.confidence = confidence or get_confidence_service(self.settings)
        self.evidence = get_evidence_service(self.settings)

    # ------------------------------------------------------------ artifacts
    def register_artifact_files(
        self,
        session,
        analysis: Analysis,
        files: list[dict[str, Any]],
        *,
        default_synthetic: bool = False,
    ) -> dict[str, str]:
        """Copy pipeline-produced files into artifact storage and register rows.

        Accepts ``{"name", "path" | "data", "kind", "source", "bounds", "crs"}`` per
        entry and returns a name/id -> serving-URL map for evidence rewriting. Files
        are copied, never moved, so the pipeline keeps its working directory.
        """
        urls: dict[str, str] = {}
        analysis_id = analysis.id
        for item in files:
            name = safe_filename(str(item.get("name") or "artifact.bin"), fallback="artifact.bin")
            kind = str(item.get("kind") or _kind_from_name(name))
            existing = next((row for row in list_artifacts(session, analysis_id) if row.name == name), None)
            artifact_id = existing.id if existing else new_id("artifact")
            target = self.store.resolve(_storage_kind(kind), analysis_id, name)
            target.parent.mkdir(parents=True, exist_ok=True)

            source_path = item.get("path")
            data = item.get("data")
            if source_path:
                shutil.copyfile(Path(source_path), target)
            elif data is not None:
                target.write_bytes(data if isinstance(data, bytes) else str(data).encode("utf-8"))
            elif existing:
                pass  # already stored; reuse the row
            else:
                logger.warning("artifact %s for %s has neither path nor data", name, analysis_id)
                continue

            size = target.stat().st_size if target.exists() else int(item.get("size_bytes") or 0)
            digest = item.get("sha256") or _sha256(target) if target.exists() else None
            url = self._artifact_url(analysis_id, artifact_id)

            payload = dict(item)
            payload.update(
                {
                    "id": artifact_id,
                    "analysis_id": analysis_id,
                    "name": name,
                    "kind": kind,
                    "relative_path": self.store.relative(target),
                    "media_type": str(item.get("media_type") or media_type_for(name)),
                    "size_bytes": size,
                    "sha256": digest,
                    "source": item.get("source"),
                    "bounds": _normalise_bounds(item.get("bounds")),
                    "crs": item.get("crs"),
                    "synthetic": bool(item.get("synthetic", default_synthetic)),
                }
            )
            artifact = Artifact(
                id=artifact_id,
                analysis_id=analysis_id,
                name=name,
                kind=kind,
                relative_path=payload["relative_path"],
                media_type=payload["media_type"],
                size_bytes=payload["size_bytes"],
                sha256=payload["sha256"],
                source=payload.get("source"),
                bounds=payload["bounds"],
                crs=payload.get("crs"),
                synthetic=payload["synthetic"],
                meta={
                    key: value
                    for key, value in item.items()
                    if key in {"model", "preprocessing_version", "bands", "class_map_version", "tile", "description"}
                }
                or None,
            )
            register_artifact(session, artifact)
            urls[name] = url
            urls[artifact_id] = url
        return urls

    # ------------------------------------------------------------- finalize
    def finalize(
        self,
        session,
        analysis: Analysis,
        outcome: dict[str, Any],
        *,
        validation: ValidationResponse | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate evidence, apply the confidence policy and store the result."""
        validation_payload = _as_dict(validation)
        georef_allowed = bool(
            validation_payload.get("geographic_fields_allowed", validation_payload.get("georeferenced", False))
        )
        pipeline_mode = str(outcome.get("pipeline_mode") or analysis.pipeline_mode or "python")
        synthetic = pipeline_mode == "stub"

        # Scene bounds decide coarse location labels.
        scene_bounds = _scene_bounds(analysis, validation_payload)
        measurement_crs = _measurement_crs(validation_payload)

        files = list(outcome.get("artifacts") or [])
        artifact_urls = self.register_artifact_files(session, analysis, files, default_synthetic=synthetic)

        raw_warnings = [Warning.model_validate(item) if isinstance(item, dict) else item for item in outcome.get("warnings") or []]

        evidence: Evidence | None = None
        if outcome.get("evidence") is not None:
            evidence = self.evidence.normalise(
                outcome.get("evidence"),
                analysis_id=analysis.id,
                georeferenced=georef_allowed,
                scene_bounds=scene_bounds,
                measurement_crs=measurement_crs,
                artifact_urls=artifact_urls,
                synthetic=synthetic,
            )
            raw_warnings.extend(self.evidence.check_overlay_shape(evidence))

        routing = analysis.routing or {}
        confidence = self.confidence.evaluate(
            task=analysis.task,
            specialists=list(outcome.get("specialists") or []),
            evidence=evidence,
            extra_warnings=[
                Warning.model_validate(item) if isinstance(item, dict) else item
                for item in (outcome.get("confidence_warnings") or [])
            ],
        )
        raw_warnings.extend(confidence.warnings)

        answer = outcome.get("answer")
        answer_type = outcome.get("answer_type")
        if confidence.decision == ConfidenceDecision.ABSTAINED:
            # Section 11.2/12.3: below the abstention threshold the system says it
            # cannot answer reliably instead of passing on a model's guess.
            answer = self.confidence.abstain_text(confidence)
            answer_type = AnswerType.ABSTAINED.value
        elif not answer:
            answer = self.evidence.summarise(evidence, task=analysis.task, question=analysis.question)
        if answer and not answer_type:
            answer_type = _answer_type_for(analysis.task)
        if not answer and confidence.decision != ConfidenceDecision.ABSTAINED:
            raw_warnings.append(
                Warning(
                    code="NO_COMPOSED_ANSWER",
                    level=WarningLevel.WARNING,
                    message="No specialist produced an answer and no evidence-backed template sentence "
                    "was available, so the response is empty rather than invented.",
                )
            )

        models = _merge_models(routing.get("models"), outcome.get("models"))
        trace = list(analysis.trace or []) + list(routing.get("trace") or []) + list(outcome.get("trace") or [])
        if confidence.decision == ConfidenceDecision.ABSTAINED:
            trace.append("abstained_by_confidence_policy")
        else:
            trace.append("calibrated_confidence")
        trace.append("composed_answer" if answer else "no_answer_composed")

        warnings_payload = _merge_warnings(
            [Warning.model_validate(item) if isinstance(item, dict) else item for item in (analysis.warnings or [])],
            [Warning.model_validate(item) if isinstance(item, dict) else item for item in (routing.get("warnings") or [])],
            raw_warnings,
            evidence.warnings if evidence else [],
        )

        result = {
            "analysis_id": analysis.id,
            "status": AnalysisStatus.COMPLETED.value,
            "input_interpretation": {
                "detected_input_type": analysis.input_type,
                "detected_modalities": analysis.modalities or [],
                "file_roles": analysis.roles,
                "intent": analysis.intent,
                "rationale": (validation_payload.get("interpretation") or {}).get("rationale") or [],
                "certainty": (validation_payload.get("interpretation") or {}).get("certainty"),
            },
            "task": analysis.task,
            "answer": answer,
            "answer_type": answer_type,
            "evidence": evidence.model_dump(mode="json") if evidence else None,
            "confidence": confidence.model_dump(mode="json"),
            "models": [model.model_dump(mode="json") for model in models],
            "warnings": [warning.model_dump(mode="json") for warning in warnings_payload],
            "execution_trace": _dedupe_strings(trace),
            "artifacts": [link.model_dump(mode="json") for link in self.artifact_links(session, analysis.id)],
            "validation": _validation_snapshot(validation_payload),
            "versions": _versions(outcome, self.confidence, self.settings),
            "pipeline": {
                "mode": pipeline_mode,
                "callable": outcome.get("pipeline_callable"),
                "authoritative": pipeline_mode == "python",
                "worker": outcome.get("worker"),
                "note": outcome.get("note"),
            },
            "disclaimer": STUB_DISCLAIMER if synthetic else outcome.get("disclaimer"),
            "question": analysis.question,
            "upload_ids": analysis.upload_ids,
        }
        result["links"] = self.links(analysis.id)

        add_warnings(session, analysis, [warning.model_dump() for warning in warnings_payload], commit=False)
        append_trace(session, analysis, _dedupe_strings(trace), commit=False)
        complete(session, analysis, result=result)
        logger.info(
            "analysis finished id=%s task=%s decision=%s artifacts=%s",
            analysis.id,
            analysis.task,
            confidence.decision,
            len(result["artifacts"]),
        )
        return result

    # --------------------------------------------------------------- read
    def links(self, analysis_id: str) -> dict[str, str]:
        prefix = self.settings.api_prefix
        return {
            "self": f"{prefix}/analyses/{analysis_id}",
            "status": f"{prefix}/analyses/{analysis_id}/status",
            "result": f"{prefix}/analyses/{analysis_id}/result",
            "report": f"{prefix}/analyses/{analysis_id}/report",
            "artifacts": f"{prefix}/analyses/{analysis_id}/artifacts",
        }

    def artifact_links(self, session, analysis_id: str) -> list[ArtifactLink]:
        return [
            ArtifactLink(
                artifact_id=artifact.id,
                name=artifact.name,
                kind=artifact.kind,
                url=self._artifact_url(analysis_id, artifact.id),
                media_type=artifact.media_type,
                size_bytes=artifact.size_bytes,
                source=artifact.source,
                bounds=artifact.bounds,
                crs=artifact.crs,
                synthetic=artifact.synthetic,
                sha256=artifact.sha256,
            )
            for artifact in list_artifacts(session, analysis_id)
        ]

    def build_result(self, session, analysis: Analysis) -> AnalysisResult:
        """Rebuild the public result, refusing to invent one that does not exist."""
        stored = analysis.result
        if analysis.status == AnalysisStatus.QUEUED.value or analysis.status == AnalysisStatus.RUNNING.value:
            raise Conflict(
                "The analysis has not finished yet; poll the status endpoint.",
                detail={"analysis_id": analysis.id, "status": analysis.status, "stage": analysis.stage},
            )
        if analysis.status == AnalysisStatus.NEEDS_CLARIFICATION.value:
            raise Conflict(
                "The analysis is waiting for clarification, so there is no result yet.",
                detail={"analysis_id": analysis.id, "clarification": analysis.clarification},
            )
        if analysis.status == AnalysisStatus.FAILED.value or not stored:
            raise Conflict(
                analysis.error_message or "The analysis failed, so there is no result.",
                detail={
                    "analysis_id": analysis.id,
                    "code": analysis.error_code,
                    "status": analysis.status,
                    "detail": analysis.error_detail,
                },
            )
        payload = dict(stored)
        payload["artifacts"] = [link.model_dump(mode="json") for link in self.artifact_links(session, analysis.id)]
        payload["links"] = self.links(analysis.id)
        payload["created_at"] = analysis.created_at
        payload["finished_at"] = analysis.finished_at
        payload["duration_seconds"] = analysis.duration_seconds
        payload["status"] = analysis.status
        return AnalysisResult.model_validate(payload)

    def _artifact_url(self, analysis_id: str, artifact_id: str) -> str:
        return f"{self.settings.api_prefix}/analyses/{analysis_id}/artifacts/{artifact_id}"


# ------------------------------------------------------------------ helpers
def _as_dict(value: ValidationResponse | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, ValidationResponse):
        return value.model_dump(mode="json")
    return dict(value)


def _scene_bounds(analysis: Analysis, validation: dict[str, Any]) -> list[list[float]] | None:
    for report in validation.get("files") or []:
        metadata = (report or {}).get("metadata") or {}
        bounds = metadata.get("bounds")
        if bounds:
            return bounds
    extra = validation.get("extra") or {}
    return extra.get("bounds")


def _measurement_crs(validation: dict[str, Any]) -> str | None:
    for report in validation.get("files") or []:
        metadata = (report or {}).get("metadata") or {}
        if metadata.get("measurement_crs"):
            return str(metadata["measurement_crs"])
    return None


def _validation_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "valid",
        "detected_input_type",
        "detected_modalities",
        "crs",
        "aligned",
        "overlap_percentage",
        "routing_candidates",
        "georeferenced",
        "geographic_fields_allowed",
        "validation_version",
    )
    snapshot = {key: payload.get(key) for key in keys if key in payload}
    snapshot["errors"] = payload.get("errors") or []
    snapshot["warnings"] = payload.get("warnings") or []
    pair = payload.get("pair") or {}
    if pair:
        snapshot["pair"] = {
            "valid": pair.get("valid"),
            "overlap_percentage": pair.get("overlap_percentage"),
            "crs_compatible": pair.get("crs_compatible"),
            "aligned": pair.get("aligned"),
            "alignment_tolerance_pixels": pair.get("alignment_tolerance_pixels"),
            "temporal_order": pair.get("temporal_order"),
            "before_upload_id": pair.get("before_upload_id"),
            "after_upload_id": pair.get("after_upload_id"),
        }
    return snapshot


def _versions(outcome: dict[str, Any], confidence: ConfidenceService, settings: Settings) -> dict[str, Any]:
    raw = dict(outcome.get("versions") or {})
    raw.setdefault("code", settings.version)
    raw.setdefault("thresholds", confidence.version)
    return raw


def _merge_models(routing_models: Any, outcome_models: Any) -> list[ModelRef]:
    merged: dict[str, ModelRef] = {}
    for source in (routing_models or [], outcome_models or []):
        for item in source:
            try:
                model = ModelRef.model_validate(item) if isinstance(item, dict) else item
            except Exception:  # noqa: BLE001 - a bad provenance entry must not lose the result
                continue
            key = model.internal_name or model.name
            merged.setdefault(key, model)
    return list(merged.values())


def _merge_warnings(*groups: list[Warning]) -> list[Warning]:
    seen: set[tuple[str, str]] = set()
    result: list[Warning] = []
    for group in groups:
        for warning in group or []:
            key = (warning.code, warning.message)
            if key in seen:
                continue
            seen.add(key)
            result.append(warning)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _normalise_bounds(bounds: Any) -> list[list[float]] | None:
    if not bounds:
        return None
    if isinstance(bounds[0], (list, tuple)):
        return [[float(v) for v in pair] for pair in bounds]
    if len(bounds) == 4:
        west, south, east, north = (float(value) for value in bounds)
        return [[south, west], [north, east]]
    return None


def _kind_from_name(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith((".geojson", ".json")):
        return "vector"
    if lowered.endswith((".png", ".jpg", ".jpeg")):
        return "overlay"
    if lowered.endswith((".tif", ".tiff")):
        return "mask"
    if lowered.endswith((".html", ".pdf")):
        return "report"
    return "other"


def _storage_kind(artifact_kind: str) -> str:
    return "geojson" if artifact_kind in {"vector", "geojson"} else ("reports" if artifact_kind == "report" else "masks")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _answer_type_for(task: str | None) -> str | None:
    return {
        Task.SINGLE_SCENE_VQA.value: AnswerType.SCENE_DESCRIPTION.value,
        Task.BI_TEMPORAL_CHANGE.value: AnswerType.CHANGE_SUMMARY.value,
        Task.OPTICAL_SAR_LAND_COVER.value: AnswerType.LAND_COVER_SUMMARY.value,
    }.get(str(task))


_service: ResultService | None = None


def get_result_service(settings: Settings | None = None) -> ResultService:
    global _service
    if _service is None:
        _service = ResultService(settings)
    return _service


def reset_result_service() -> None:
    global _service
    _service = None
