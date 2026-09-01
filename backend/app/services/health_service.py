"""Health aggregation for the FastAPI control plane."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from app.core.config import PipelineMode, QueueMode, Settings, get_settings
from app.core.storage import ArtifactStore, get_store
from app.db.repo import kv_get
from app.geospatial.raster_probe import describe_capabilities
from app.schemas.common import Warning, WarningLevel
from app.schemas.health import ComponentHealth, HealthResponse
from app.services.model_registry import ModelRegistry, get_registry

_STARTED_AT = time.monotonic()
_STALE_HEARTBEAT_SECONDS = 90.0


class HealthService:
    def __init__(
        self,
        settings: Settings | None = None,
        store: ArtifactStore | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store(self.settings)
        self.registry = registry or get_registry(self.settings)

    def check(self, session, *, queue: Any | None = None) -> HealthResponse:
        components: dict[str, ComponentHealth] = {}
        warnings: list[Warning] = []

        components["database"] = self._database(session)
        components["storage"] = self._storage()
        components["queue"] = self._queue(queue)
        components["worker"] = self._worker(session)
        components["pipeline"] = self._pipeline()
        components["geospatial"] = self._geospatial()
        components["registry"] = self._registry()

        if self.settings.queue_mode is QueueMode.INLINE:
            warnings.append(
                Warning(
                    code="INLINE_QUEUE_DEVELOPMENT_ONLY",
                    level=WarningLevel.WARNING,
                    message="The inline queue is a single-process development fallback, not the production GPU path.",
                )
            )

        response = HealthResponse(
            version=self.settings.version,
            environment=self.settings.env.value,
            time=datetime.now(timezone.utc),
            uptime_seconds=round(time.monotonic() - _STARTED_AT, 3),
            components=components,
            warnings=warnings,
            limits={
                "allowed_extensions": sorted(self.settings.allowed_extensions),
                "max_upload_bytes": self.settings.max_upload_bytes,
                "max_decompressed_bytes": self.settings.max_decompressed_bytes,
                "max_raster_pixels": self.settings.max_raster_pixels,
                "max_files_per_analysis": self.settings.max_files_per_analysis,
                "min_overlap_percent": self.settings.min_overlap_percent,
                "max_residual_offset_pixels": self.settings.max_residual_offset_pixels,
            },
        )
        return response.with_overall()

    def _database(self, session) -> ComponentHealth:
        try:
            session.execute(text("SELECT 1"))
            return ComponentHealth(name="database", status="ok", message="Metadata database is reachable.")
        except Exception as exc:  # noqa: BLE001 - health must report, not raise
            return ComponentHealth(name="database", status="error", message="Metadata database is unavailable.", detail={"error": str(exc)})

    def _storage(self) -> ComponentHealth:
        root = self.store.root
        writable = root.is_dir() and os.access(root, os.R_OK | os.W_OK)
        return ComponentHealth(
            name="storage",
            status="ok" if writable else "error",
            message="Artifact storage is readable and writable." if writable else "Artifact storage is not usable.",
            detail=self.store.describe(),
        )

    def _queue(self, queue: Any | None) -> ComponentHealth:
        if queue is None:
            return ComponentHealth(name="queue", status="unknown", message="Queue backend has not been initialized.")
        try:
            detail = queue.health()
        except Exception as exc:  # noqa: BLE001
            return ComponentHealth(name="queue", status="error", message="Queue health check failed.", detail={"error": str(exc)})
        status = str(detail.pop("status", "unknown"))
        message = str(detail.pop("message", "Queue backend status reported."))
        return ComponentHealth(name="queue", status=status, message=message, detail=detail)

    def _worker(self, session) -> ComponentHealth:
        heartbeat = kv_get(session, "worker:heartbeat")
        if not isinstance(heartbeat, dict):
            status = "degraded" if self.settings.queue_mode is QueueMode.INLINE else "unknown"
            return ComponentHealth(name="worker", status=status, message="No worker heartbeat has been recorded yet.")
        raw_at = heartbeat.get("at")
        try:
            moment = datetime.fromisoformat(str(raw_at).replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            age = max(0.0, (datetime.now(timezone.utc) - moment).total_seconds())
        except (TypeError, ValueError):
            age = None
        stale = age is None or age > _STALE_HEARTBEAT_SECONDS
        active = kv_get(session, "worker:active")
        if isinstance(active, dict):
            try:
                active_at = datetime.fromisoformat(str(active.get("at")).replace("Z", "+00:00"))
                if active_at.tzinfo is None:
                    active_at = active_at.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - active_at).total_seconds() <= _STALE_HEARTBEAT_SECONDS:
                    heartbeat = {**heartbeat, "state": "busy", "analysis_id": active.get("analysis_id")}
            except (TypeError, ValueError):
                pass
        return ComponentHealth(
            name="worker",
            status="degraded" if stale else "ok",
            message="Worker heartbeat is stale." if stale else "Worker heartbeat is current.",
            age_seconds=round(age, 3) if age is not None else None,
            detail=heartbeat,
        )

    def _pipeline(self) -> ComponentHealth:
        mode = self.settings.pipeline_mode
        if mode is PipelineMode.PYTHON:
            from app.workers.pipeline import probe_pipeline

            detail = probe_pipeline(self.settings)
            return ComponentHealth(
                name="pipeline",
                status="ok" if detail.get("available") else "error",
                message=str(detail.get("message") or "Python pipeline configured."),
                detail=detail,
            )
        message = (
            "No inference pipeline is attached; analyses fail safely."
            if mode is PipelineMode.UNATTACHED
            else "Stub mode is active; results are non-authoritative."
        )
        return ComponentHealth(name="pipeline", status="degraded", message=message, detail={"mode": mode.value})

    def _geospatial(self) -> ComponentHealth:
        detail = describe_capabilities()
        backends = detail.get("backends") or {}
        reduced = not bool(backends.get("rasterio")) or not bool(backends.get("pyproj"))
        return ComponentHealth(
            name="geospatial",
            status="degraded" if reduced else "ok",
            message="Reduced geospatial capability is active." if reduced else "Full geospatial extras are available.",
            detail=detail,
        )

    def _registry(self) -> ComponentHealth:
        snapshot = self.registry.snapshot(probe=self.settings.pipeline_mode is PipelineMode.PYTHON)
        if self.registry.load_error:
            status = "error"
        elif not snapshot.models:
            status = "error"
        elif self.settings.pipeline_mode is PipelineMode.PYTHON and not snapshot.inference_ready:
            status = "degraded"
        else:
            status = "ok"
        return ComponentHealth(
            name="registry",
            status=status,
            message="Model registry loaded." if status != "error" else "Model registry is unavailable.",
            detail={
                "version": snapshot.registry_version,
                "entries": len(snapshot.models),
                "inference_ready": snapshot.inference_ready,
                "path": snapshot.path,
            },
        )


_service: HealthService | None = None


def get_health_service(settings: Settings | None = None) -> HealthService:
    global _service
    if _service is None:
        _service = HealthService(settings=settings)
    return _service


def reset_health_service() -> None:
    global _service
    _service = None
