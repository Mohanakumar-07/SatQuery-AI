"""Common specialist-result contract (Implementation_Plan_v1.2.md "Model Output Contract").

Both ChangeNet and SAR-FuseSeg return this same shape so a future router/evidence
engine can treat them uniformly. This module only defines the contract; it does not
build the final agentic router (explicitly out of scope for this phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    name: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass
class SpecialistResult:
    task: str
    status: str  # "success" | "error"
    model: ModelInfo
    prediction: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status,
            "model": self.model.to_dict(),
            "prediction": self.prediction,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }
