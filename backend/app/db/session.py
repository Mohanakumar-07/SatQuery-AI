"""Database engine and session handling (SQLite for the MVP, PostgreSQL later).

Engines are cached per URL. SQLite gets WAL plus a busy timeout because the API
process and the inline worker thread write concurrently; that is the only reason
the pragmas are here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from app.core.config import get_settings
from app.db.base import Base

_engines: dict[str, Engine] = {}
_session_factories: dict[str, sessionmaker[Session]] = {}


def _prepare_engine(url: str) -> Engine:
    normalized = url
    connect_args: dict[str, object] = {}
    pool_kwargs: dict[str, object] = {}

    if normalized.startswith("sqlite"):
        raw = normalized.split("sqlite:///")[-1]
        if raw and raw != ":memory:":
            Path(raw).expanduser().parent.mkdir(parents=True, exist_ok=True)
        connect_args["check_same_thread"] = False
        # A file database is shared between the API and worker threads, so use a
        # throwaway pool plus WAL/busy-timeout pragmas; a memory database must stay
        # pinned to one connection or it disappears between sessions.
        pool_kwargs["poolclass"] = StaticPool if raw == ":memory:" or not raw else NullPool

    engine = create_engine(
        normalized,
        future=True,
        connect_args=connect_args,
        **({"pool_pre_ping": True} if not normalized.startswith("sqlite") else {}),
        **pool_kwargs,
    )

    if normalized.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


def get_engine(url: str | None = None) -> Engine:
    resolved = url or get_settings().database_url
    if resolved not in _engines:
        _engines[resolved] = _prepare_engine(resolved)
    return _engines[resolved]


def get_session_factory(url: str | None = None) -> sessionmaker[Session]:
    resolved = url or get_settings().database_url
    if resolved not in _session_factories:
        _session_factories[resolved] = sessionmaker(
            bind=get_engine(resolved),
            autoflush=False,
            expire_on_commit=False,
            class_=Session,
        )
    return _session_factories[resolved]


def init_db(url: str | None = None) -> None:
    """Create every table that does not exist yet. Safe to call repeatedly."""
    # Register every mapped table before reading ``Base.metadata``. Importing this
    # module directly (CLI/bootstrap scripts) must be as reliable as importing the
    # full FastAPI route graph first.
    from app.db import models as _models  # noqa: F401

    engine = get_engine(url)
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Transactional scope used by workers and scripts (not request handlers)."""
    session = get_session_factory(url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def reset_engines() -> None:
    """Dispose and forget cached engines (tests swap databases at runtime)."""
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()
    _session_factories.clear()
