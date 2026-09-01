"""Health and capability models (plan section 7.2 "Health", section 7.1 "Health service")."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import ResponseModel, Warning

ComponentStatus = Literal["ok", "degraded", "error", "unknown"]

_OVERALL_ORDER = {"ok": 0, "degraded": 1, "unknown": 2, "error": 3}


class ComponentHealth(ResponseModel):
    name: str
    status: ComponentStatus = "unknown"
    message: str | None = None
    #: Optional seconds-since-heartbeat for stateful components such as the worker.
    age_seconds: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(ResponseModel):
    status: ComponentStatus = "ok"
    version: str
    environment: str
    time: datetime
    uptime_seconds: float | None = None
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
    warnings: list[Warning] = Field(default_factory=list)
    #: Flattened limits the frontend needs to mirror (upload size, allowed types...).
    limits: dict[str, Any] = Field(default_factory=dict)

    def with_overall(self) -> "HealthResponse":
        worst: ComponentStatus = "ok"
        for component in self.components.values():
            if _OVERALL_ORDER[component.status] > _OVERALL_ORDER[worst]:
                worst = component.status
        self.status = worst
        return self


class RootResponse(ResponseModel):
    name: str
    version: str
    environment: str
    docs_url: str | None = None
    openapi_url: str | None = None
    health_url: str
    api_prefix: str
