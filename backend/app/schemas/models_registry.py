"""Model registry contract (plan sections 4.2, 9 and 16).

Licence fields are part of the schema, not an afterthought: section 16 requires the
terms of every checkpoint and every code repository to be recorded and verified
before packaging. Defaults are ``None``/``False`` so an unverified entry is visible in
``GET /models`` rather than looking approved.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import ResponseModel, Task, Warning

ModelStatus = Literal["available", "unavailable", "not_implemented", "loading", "error"]


class LicenceRecord(ResponseModel):
    source: str | None = None
    version: str | None = None
    code_repository: str | None = None
    code_licence: str | None = None
    checkpoint_source: str | None = None
    checkpoint_licence: str | None = None
    checksum: str | None = None
    redistribution_allowed: bool | None = None
    attribution: str | None = None
    intended_use_restrictions: str | None = None
    #: Only true once a human has read the exact terms of the downloaded artifact.
    verified: bool = False
    reviewed_at: str | None = None
    notes: str | None = None


class ModelInfo(ResponseModel):
    internal_name: str = Field(description="SatVLM | ChangeNet | SAR-FuseSeg")
    model_name: str = Field(description="Concrete checkpoint, e.g. Qwen2.5-VL-7B-Instruct.")
    version: str | None = None
    role: str = Field(description="primary_specialist | answer_composition | evidence_interpreter")
    status: ModelStatus = "not_implemented"
    #: Whether the router may select this model at all (plan section 9 permitted list).
    permitted: bool = True
    tasks: list[Task | str] = Field(default_factory=list)
    preprocessing_version: str | None = None
    calibration_version: str | None = None
    adapter_ref: str | None = Field(default=None, description="Dotted path the worker imports for this adapter.")
    requires_gpu: bool = True
    resident: bool = Field(default=False, description="Keep loaded between jobs (plan section 7.0 GPU policy).")
    loaded: bool | None = None
    last_checked_at: datetime | None = None
    licence: LicenceRecord = Field(default_factory=LicenceRecord)
    notes: str | None = None
    warnings: list[Warning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _flag_unverified_licence(self) -> "ModelInfo":
        if self.permitted and not self.licence.verified:
            self.warnings = [
                *self.warnings,
                Warning(
                    code="LICENCE_UNVERIFIED",
                    level="warning",  # normalised by Warning's enum coercion
                    message=(
                        f"Checkpoint and code licence terms for {self.internal_name} have not been "
                        "recorded and verified (plan section 16)."
                    ),
                ),
            ]
        return self

    @property
    def is_usable(self) -> bool:
        return self.permitted and self.status == "available"


class RegistryResponse(ResponseModel):
    registry_version: str
    generated_at: datetime | None = None
    path: str | None = None
    models: list[ModelInfo] = Field(default_factory=list)
    #: Internal names the router may currently select.
    permitted_models: list[str] = Field(default_factory=list)
    unavailable_models: list[str] = Field(default_factory=list)
    #: False when nothing is usable, so the UI can warn before queueing a doomed job.
    inference_ready: bool = False
    warnings: list[Warning] = Field(default_factory=list)
    pipeline_mode: str | None = None
