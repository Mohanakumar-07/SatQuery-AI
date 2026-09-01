"""Declarative base, shared column types and UTC time helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Naive UTC now.

    SQLite has no timezone-aware storage, so every timestamp in this schema is
    naive UTC by convention. The API layer re-attaches ``+00:00`` when serialising
    (see :func:`as_utc`); never call ``datetime.now()`` for a stored value.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def as_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to a naive stored timestamp for response models."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class Base(DeclarativeBase):
    """Common declarative base. Every model sets an explicit ``__tablename__``."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)


class IntKeyMixin:
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
