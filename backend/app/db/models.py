"""SQLAlchemy tables for uploads, analyses, artifacts, events and worker state.

JSON columns hold response payloads verbatim. The MVP needs ad-hoc, versioned
result fields (warnings, traces, evidence) that would otherwise need a migration per
change; relational integrity is kept only where the API actually queries — by ID and
by creation order.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, utcnow


class Upload(TimestampMixin, Base):
    """One stored client file plus whatever metadata could be probed from it."""

    __tablename__ = "uploads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_name: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)

    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    extension: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    declared_media_type: Mapped[str | None] = mapped_column(String(120))
    detected_media_type: Mapped[str] = mapped_column(String(120), nullable=False)
    #: geotiff | tiff | png | jpeg | unsupported
    media_kind: Mapped[str] = mapped_column(String(24), nullable=False, index=True)

    #: ok | error | unsupported
    probe_status: Mapped[str] = mapped_column(String(16), nullable=False, default="unsupported")
    #: None means "not georeferenced or unknown", never "assume WGS84".
    georeferenced: Mapped[bool | None] = mapped_column(Boolean)
    crs: Mapped[str | None] = mapped_column(String(64))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    band_count: Mapped[int | None] = mapped_column(Integer)
    #: ISO-8601 acquisition date when the file or a sidecar exposed one.
    acquisition_date: Mapped[str | None] = mapped_column(String(32), index=True)
    sensor: Mapped[str | None] = mapped_column(String(80))
    modality: Mapped[str | None] = mapped_column(String(16))
    probe: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    probe_error: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    #: Provenance of the metadata (file signature, GeoTIFF keys, rasterio, client hint).
    metadata_source: Mapped[str | None] = mapped_column(String(48))

    analyses: Mapped[list["Analysis"]] = relationship(
        secondary="analysis_upload",
        viewonly=True,
        lazy="noload",
        doc="Analyses that referenced this upload (view-only convenience).",
    )


class AnalysisUpload(Base):
    """Membership link so an analysis can list its uploads in client order."""

    __tablename__ = "analysis_upload"
    __table_args__ = (UniqueConstraint("analysis_id", "upload_id", name="uq_analysis_upload"),)

    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), primary_key=True
    )
    upload_id: Mapped[str] = mapped_column(ForeignKey("uploads.id"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: before | after | optical | sar | single | unknown
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="unknown")


class Analysis(TimestampMixin, Base):
    """One requested analysis: its inputs, routing decision, progress and result."""

    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_status_created", "status", "created_at"),
        Index("ix_analyses_task_created", "task", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    hints: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    #: queued | running | needs_clarification | completed | failed
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued", index=True)
    stage: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(String(255))

    # ---- interpretation / validation / routing ----
    input_type: Mapped[str | None] = mapped_column(String(24))
    modalities: Mapped[list[str] | None] = mapped_column(JSON)
    task: Mapped[str | None] = mapped_column(String(40))
    intent: Mapped[str | None] = mapped_column(String(40))
    routing: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    validation: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    clarification: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # ---- outcome ----
    answer: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    confidence: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    models: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    warnings: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    trace: Mapped[list[str] | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(40))
    error_message: Mapped[str | None] = mapped_column(Text)
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # ---- execution bookkeeping ----
    queue_backend: Mapped[str | None] = mapped_column(String(16))
    queue_job_id: Mapped[str | None] = mapped_column(String(64))
    worker_name: Mapped[str | None] = mapped_column(String(80))
    pipeline_mode: Mapped[str | None] = mapped_column(String(16))

    #: Number of times this analysis has been queued (guard against duplicate enqueue).
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    #: Wall-clock execution time in seconds, set when the analysis reaches a terminal state.
    duration_seconds: Mapped[float | None] = mapped_column(Float)

    upload_links: Mapped[list[AnalysisUpload]] = relationship(
        "AnalysisUpload",
        lazy="selectin",
        order_by="AnalysisUpload.position",
        cascade="all, delete-orphan",
        backref="analysis",
    )

    @property
    def upload_ids(self) -> list[str]:
        """Upload identifiers in the order the client submitted them."""
        return [link.upload_id for link in self.upload_links]

    @property
    def roles(self) -> dict[str, str]:
        """Resolved ``upload_id -> role`` mapping for this analysis."""
        return {link.upload_id: link.role for link in self.upload_links}


class Artifact(TimestampMixin, Base):
    """A file produced by an analysis, addressable only through its analysis."""

    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("analysis_id", "name", name="uq_artifact_analysis_name"),
        Index("ix_artifacts_analysis_kind", "analysis_id", "kind"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: overlay | mask | vector | report | scene | thumbnail | other
    kind: Mapped[str] = mapped_column(String(24), nullable=False, default="other")
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    media_type: Mapped[str] = mapped_column(String(120), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64))
    #: Which specialist or service produced it (provenance, plan section 12.0).
    source: Mapped[str | None] = mapped_column(String(80))
    #: Geographic bounds [[south, west], [north, east]] — only when georeferenced.
    bounds: Mapped[list[Any] | None] = mapped_column(JSON)
    crs: Mapped[str | None] = mapped_column(String(64))
    #: Pixel-space bounds for ungeoreferenced inputs, never geographic ones.
    pixel_extent: Mapped[list[Any] | None] = mapped_column(JSON)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    #: True when the API generated this file for wiring tests, not from a model.
    synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    analysis: Mapped[Analysis] = relationship(backref="artifacts")


class AnalysisEvent(Base):
    """Append-only progress log; the source for the polling status feed."""

    __tablename__ = "analysis_events"
    __table_args__ = (Index("ix_events_analysis_at", "analysis_id", "at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    status: Mapped[str | None] = mapped_column(String(24))
    stage: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str | None] = mapped_column(String(255))
    data: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    analysis: Mapped[Analysis] = relationship(backref="events")


class KeyValue(Base):
    """Small shared-state table: worker heartbeats, active job, feature flags."""

    __tablename__ = "kv_store"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
