"""Specialist adapter contract and live availability probing.

**Owned by the ML team.** The web layer only ever talks to a model through this
interface, which is how plan section 9's rule "fall back or abstain when a required
specialist is unavailable" is enforced: ``probe_adapter`` never raises, so an import
error, a missing GPU or an absent checkpoint becomes a registry status the router can
act on.

Implementors fill in ``app/models/satvlm_adapter.py``, ``changenet_adapter.py`` and
``sar_fuseseg_adapter.py``. Nothing in this module performs inference.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.core.logging import get_logger
from app.preprocessing.base import NotImplementedInContract, SceneBundle  # noqa: F401  (re-exported contract)
from app.schemas.common import Warning, WarningLevel

logger = get_logger("models.base")


@dataclass
class AdapterProbe:
    """Result of asking an adapter whether it could run right now."""

    available: bool = False
    status: str = "not_implemented"  # available | unavailable | not_implemented | error
    reason: str | None = None
    code: str | None = None
    warnings: list[Warning] = field(default_factory=list)
    loaded: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterRequest:
    """Everything a specialist receives: the canonical bundle plus its own work dir."""

    analysis_id: str
    task: str
    question: str
    bundle: SceneBundle
    work_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    preprocessing_version: str | None = None


@dataclass
class AdapterResponse:
    """Raw specialist output, before the evidence engine turns it into facts.

    Masks must be written to ``work_dir`` and referenced by path: the API never
    serialises tensors (plan section 12.0).
    """

    source: str
    version: str | None = None
    #: Paths of masks/overlays written under work_dir.
    mask_paths: list[Path] = field(default_factory=list)
    overlay_paths: list[Path] = field(default_factory=list)
    geojson_paths: list[Path] = field(default_factory=list)
    #: Model-specific score whose meaning is declared by ``score_kind``.
    raw_score: float | None = None
    score_kind: str = "unknown"
    #: Structured facts from the specialist, e.g. binary change facts.
    facts: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    answer_status: str | None = None
    evidence_coverage: float | None = None
    unsupported_claims: int | None = None
    trace: list[str] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class SpecialistAdapter(Protocol):
    """The minimum surface an adapter must provide."""

    internal_name: str
    model_name: str
    version: str
    preprocessing_version: str

    def available(self, *, capabilities: dict[str, bool] | None = None) -> AdapterProbe:
        """Cheap, non-loading check. Must not import torch or touch a GPU."""
        ...

    def load(self, *, device: str | None = None) -> None:
        """Load weights once; safe to call repeatedly."""
        ...

    def infer(self, request: AdapterRequest) -> AdapterResponse:
        """Run one inference on the prepared scene."""
        ...

    def unload(self) -> None:
        """Release GPU memory (plan section 7.0 memory policy)."""
        ...


class BaseSpecialistAdapter:
    """Convenience base class: records identity, fails loudly until implemented."""

    internal_name = "unspecified"
    model_name = "unspecified"
    version = "unversioned"
    preprocessing_version = "unspecified"
    requires_gpu = True
    #: Optional checkpoint file whose presence gates availability.
    checkpoint_hint: str | None = None

    def __init__(self, *, settings: Any | None = None, params: dict[str, Any] | None = None) -> None:
        self.settings = settings
        self.params = dict(params or {})
        self._loaded = False

    def available(self, *, capabilities: dict[str, bool] | None = None) -> AdapterProbe:  # pragma: no cover - stub
        return AdapterProbe(
            available=False,
            status="not_implemented",
            code="ADAPTER_NOT_IMPLEMENTED",
            reason=f"{self.internal_name} adapter has not been implemented yet.",
        )

    def load(self, *, device: str | None = None) -> None:
        raise NotImplementedInContract(f"{self.internal_name}.load() is not implemented.")

    def infer(self, request: AdapterRequest) -> AdapterResponse:
        raise NotImplementedInContract(f"{self.internal_name}.infer() is not implemented for task {request.task!r}.")

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def checkpoint_path(self) -> Path | None:
        if not self.checkpoint_hint or self.settings is None:
            return None
        return Path(self.settings.checkpoints_dir) / self.checkpoint_hint


def import_object(dotted: str) -> Any:
    """Import ``package.module:attr`` or ``package.module.attr``."""
    if not dotted:
        raise ImportError("empty reference")
    ref = dotted.replace(":", ".") if ":" in dotted else dotted
    module_path, _, attribute = ref.rpartition(".")
    if not module_path:
        raise ImportError(f"{dotted!r} is not a dotted path to an object.")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(f"{module_path} has no attribute {attribute!r}") from exc


def probe_adapter(reference: str, *, capabilities: dict[str, bool] | None = None) -> AdapterProbe:
    """Import an adapter and ask it whether it could run. Never raises."""
    try:
        target = import_object(reference)
    except Exception as exc:  # noqa: BLE001 - any import failure is a status
        return AdapterProbe(
            available=False,
            status="unavailable",
            code="ADAPTER_IMPORT_FAILED",
            reason=f"Could not import {reference!r}: {exc}",
        )

    instance: Any
    try:
        instance = target() if isinstance(target, type) else target
    except Exception as exc:  # noqa: BLE001 - constructor failures are a status too
        return AdapterProbe(
            available=False,
            status="error",
            code="ADAPTER_CONSTRUCT_FAILED",
            reason=f"{reference!r} could not be constructed: {exc}",
        )

    checker = getattr(instance, "available", None)
    if not callable(checker):
        return AdapterProbe(
            available=False,
            status="error",
            code="ADAPTER_CONTRACT_MISMATCH",
            reason=f"{reference!r} does not implement the SpecialistAdapter contract.",
        )
    try:
        probe = checker(capabilities=capabilities or {})
    except TypeError:
        probe = checker()  # adapters may ignore capabilities in simple implementations
    except Exception as exc:  # noqa: BLE001
        return AdapterProbe(
            available=False,
            status="error",
            code="ADAPTER_PROBE_FAILED",
            reason=f"{reference!r}.available() raised {type(exc).__name__}: {exc}",
        )
    if isinstance(probe, AdapterProbe):
        return probe
    return AdapterProbe(
        available=bool(probe),
        status="available" if probe else "unavailable",
        reason=None if probe else f"{reference!r}.available() returned False.",
    )


def describe_adapters() -> list[dict[str, Any]]:
    """Contract summary for ``/docs`` readers; no model is ever loaded here."""
    return [
        {
            "internal_name": name,
            "adapter": reference,
            "expects": "SceneBundle in, AdapterResponse out",
            "owned_by": owner,
        }
        for name, reference, owner in (
            ("SatVLM", "app.models.satvlm_adapter.SatVLMAdapter", "member 1"),
            ("ChangeNet", "app.models.changenet_adapter.ChangeNetAdapter", "member 3"),
            ("SAR-FuseSeg", "app.models.sar_fuseseg_adapter.SarFuseSegAdapter", "member 3"),
        )
    ]
