"""Versioned model registry and live adapter availability (plan sections 4.2, 9, 16).

The registry file is the source of truth for what the router is *allowed* to pick;
live availability comes from importing each adapter and calling ``available()``. Import
failures are recorded, never raised: a missing torch install must show up as
``status="unavailable"`` in ``GET /models`` rather than taking the API down.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.geospatial.raster_probe import capability_matrix
from app.schemas.common import Task, Warning, WarningLevel
from app.schemas.models_registry import ModelInfo, RegistryResponse

_PROBE_TTL_SECONDS = 60.0


class RegistryError(AppError):
    status_code = 500
    code = ErrorCode.INTERNAL_ERROR
    default_message = "The model registry could not be read."


class ModelRegistry:
    """Thread-safe loader for ``model_registry.json`` with cached adapter probes."""

    def __init__(self, path: str | Path | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.path = Path(path) if path else self._settings.model_registry_path
        self._lock = threading.Lock()
        self._models: list[ModelInfo] = []
        self.version: str = "unloaded"
        self._loaded_mtime: float | None = None
        self._probe_cache: dict[str, tuple[float, str, list[Warning]]] = {}
        self._load_error: str | None = None
        self._notes: str = ""
        self.load()

    # ----------------------------------------------------------------- load
    def load(self, *, force: bool = False) -> None:
        """Read the registry from disk if it changed (or when forced)."""
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime if self.path.exists() else None
            except OSError:
                mtime = None
            if not force and mtime is not None and mtime == self._loaded_mtime and self._models:
                return
            if not self.path.exists():
                self._load_error = f"registry file not found at {self.path}"
                self._models = []
                self.version = "missing"
                return
            try:
                raw: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self._load_error = f"registry JSON is invalid: {exc}"
                self._models = []
                self.version = "invalid"
                return

            models: list[ModelInfo] = []
            for entry in raw.get("models") or []:
                try:
                    models.append(ModelInfo.model_validate(entry))
                except Exception as exc:  # noqa: BLE001 - one bad entry must not hide the rest
                    self._load_error = f"registry entry {entry.get('internal_name')!r} is invalid: {exc}"
                    continue
            self._models = models
            self.version = str(raw.get("registry_version") or "unknown")
            self._notes = str(raw.get("notes") or "")
            self._load_error = None
            self._loaded_mtime = mtime
            self._probe_cache.clear()

    @property
    def models(self) -> list[ModelInfo]:
        self.load()
        return list(self._models)

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ---------------------------------------------------------------- query
    def get(self, internal_name: str) -> ModelInfo | None:
        target = internal_name.strip().lower()
        for model in self.models:
            if model.internal_name.lower() == target:
                return model
        return None

    def permitted_for(self, task: Task | str) -> list[ModelInfo]:
        """Models the router may select for a task, in registry order."""
        wanted = str(getattr(task, "value", task))
        return [model for model in self.models if model.permitted and wanted in _task_values(model)]

    def usable_for(self, task: Task | str) -> list[ModelInfo]:
        return [model for model in self.permitted_for(task) if model.is_usable]

    def composition_models(self) -> list[ModelInfo]:
        return [model for model in self.models if model.role == "answer_composition" and model.permitted]

    # -------------------------------------------------------------- probing
    def probe(self, model: ModelInfo, *, refresh: bool = False) -> tuple[str, list[Warning]]:
        """Return ``(status, warnings)`` for one registry entry, probing adapters."""
        now = time.time()
        cached = self._probe_cache.get(model.internal_name)
        if cached and not refresh and now - cached[0] < _PROBE_TTL_SECONDS:
            return cached[1], list(cached[2])

        warnings: list[Warning] = []
        if not model.permitted:
            status, warnings = "unavailable", [
                Warning(
                    code="MODEL_NOT_PERMITTED",
                    level=WarningLevel.WARNING,
                    message=f"{model.internal_name} is excluded from the permitted model list.",
                )
            ]
        elif not model.adapter_ref:
            status = "not_implemented"
            warnings = [
                Warning(
                    code="ADAPTER_NOT_IMPLEMENTED",
                    level=WarningLevel.WARNING,
                    message=f"{model.internal_name} has no adapter_ref in the registry.",
                )
            ]
        else:
            status, warnings = self._probe_adapter(model)

        with self._lock:
            self._probe_cache[model.internal_name] = (now, status, warnings)
        return status, list(warnings)

    def _probe_adapter(self, model: ModelInfo) -> tuple[str, list[Warning]]:
        from app.models.base import probe_adapter

        outcome = probe_adapter(model.adapter_ref or "", capabilities=capability_matrix())
        warnings = list(outcome.warnings)
        if not outcome.available:
            if outcome.reason:
                warnings.insert(
                    0,
                    Warning(
                        code=outcome.code or "ADAPTER_UNAVAILABLE",
                        level=WarningLevel.WARNING,
                        message=outcome.reason,
                        detail={"adapter_ref": model.adapter_ref},
                    ),
                )
            return outcome.status, warnings
        return "available", warnings

    # -------------------------------------------------------------- export
    def snapshot(self, *, probe: bool = True, refresh: bool = False) -> RegistryResponse:
        models: list[ModelInfo] = []
        for model in self.models:
            if probe:
                status, warnings = self.probe(model, refresh=refresh)
                model = model.model_copy(update={"status": status, "warnings": warnings})
            models.append(model)

        permitted = [m.internal_name for m in models if m.is_usable]
        unavailable = [m.internal_name for m in models if not m.is_usable]
        warnings: list[Warning] = []
        if self._load_error:
            warnings.append(
                Warning(
                    code="REGISTRY_UNREADABLE",
                    level=WarningLevel.ERROR,
                    message=self._load_error,
                )
            )
        if not permitted:
            warnings.append(
                Warning(
                    code="NO_MODELS_AVAILABLE",
                    level=WarningLevel.WARNING,
                    message="No specialist model is currently available; analyses will abstain or fail safely.",
                )
            )
        return RegistryResponse(
            registry_version=self.version,
            generated_at=None,
            path=str(self.path),
            models=models,
            permitted_models=permitted,
            unavailable_models=unavailable,
            inference_ready=bool(permitted),
            warnings=warnings,
            pipeline_mode=self._settings.pipeline_mode.value,
        )


def _task_values(model: ModelInfo) -> list[str]:
    return [str(getattr(task, "value", task)) for task in model.tasks]


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry(settings: Settings | None = None) -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry(settings=settings)
    return _registry


def reset_registry_cache() -> None:
    global _registry
    with _registry_lock:
        _registry = None
