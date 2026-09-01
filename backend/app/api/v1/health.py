"""Health and model-registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.v1.deps import AppSettings, DbSession, JobQueue
from app.schemas.health import HealthResponse
from app.schemas.models_registry import RegistryResponse
from app.services.health_service import get_health_service
from app.services.model_registry import get_registry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service and capability health")
def health(settings: AppSettings, session: DbSession, queue: JobQueue) -> HealthResponse:
    return get_health_service(settings).check(session, queue=queue)


@router.get("/models", response_model=RegistryResponse, summary="Permitted model registry")
def models(
    settings: AppSettings,
    refresh: bool = Query(default=False, description="Refresh cached adapter availability probes."),
) -> RegistryResponse:
    return get_registry(settings).snapshot(probe=True, refresh=refresh)
