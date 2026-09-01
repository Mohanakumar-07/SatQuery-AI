"""Structured logging with a request/analysis correlation field."""

from __future__ import annotations

import contextvars
import logging
import sys
from typing import Any

from app.core.config import get_settings

_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_CONFIGURED = False
_LOG_RECORD_FACTORY = logging.LogRecord


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.ctx = _REQUEST_ID.get()
        return True


def configure_logging(level: str | None = None, *, force: bool = False) -> None:
    """Install one concise stderr handler for the whole app.

    uvicorn already configures its own handlers; this only guarantees that
    ``app.*`` loggers emit at the configured level with the correlation field.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    settings = get_settings()
    resolved = (level or settings.log_level).upper()
    root = logging.getLogger("satquery")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(name)s] %(ctx)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        handler.addFilter(_CorrelationFilter())
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            if not any(isinstance(f, _CorrelationFilter) for f in handler.filters):
                handler.addFilter(_CorrelationFilter())
    root.setLevel(resolved)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"satquery.{name}")


def set_request_id(value: str) -> None:
    _REQUEST_ID.set(value)


def get_request_id() -> str:
    return _REQUEST_ID.get()


def log_kv(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit ``event key=value key=value`` so logs stay greppable."""
    rendered = " ".join(f"{key}={value!r}" for key, value in fields.items() if value is not None)
    logger.log(level, "%s %s", event, rendered)
