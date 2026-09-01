"""Redis/RQ production queue and a controlled inline development fallback."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from app.core.config import QueueMode, Settings, get_settings
from app.core.errors import QueueUnavailable
from app.core.ids import new_id


@dataclass(frozen=True)
class EnqueueResult:
    backend: str
    job_id: str


class InlineQueue:
    """Exactly one in-process worker thread, for development only."""

    name = "inline"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="satquery-inline-worker")
        self._lock = threading.RLock()
        self._futures: dict[str, Future[Any]] = {}
        self._closed = False

    def enqueue(self, analysis_id: str) -> EnqueueResult:
        with self._lock:
            if self._closed:
                raise QueueUnavailable("The inline queue is shutting down.")
            job_id = new_id("job")
            from app.workers.tasks import run_analysis

            future = self._executor.submit(run_analysis, analysis_id)
            self._futures[job_id] = future
            future.add_done_callback(lambda _: self._discard(job_id))
        return EnqueueResult(backend=self.name, job_id=job_id)

    def _discard(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    def health(self) -> dict[str, Any]:
        with self._lock:
            active = sum(1 for future in self._futures.values() if not future.done())
            closed = self._closed
        return {
            "status": "error" if closed else "degraded",
            "message": "Inline development queue is active." if not closed else "Inline queue is closed.",
            "backend": self.name,
            "authoritative": False,
            "active_jobs": active,
            "max_workers": 1,
        }

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=False)


class RqQueue:
    """Redis Queue adapter used by the isolated CPU/GPU worker."""

    name = "rq"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            from redis import Redis
            from rq import Queue
        except ImportError as exc:
            raise QueueUnavailable(
                "Redis/RQ dependencies are not installed.",
                detail={"install": "pip install redis rq"},
            ) from exc
        self.connection = Redis.from_url(settings.redis_url)
        self.queue = Queue("satquery", connection=self.connection, default_timeout=settings.job_timeout_seconds)

    def enqueue(self, analysis_id: str) -> EnqueueResult:
        try:
            self.connection.ping()
            job_id = new_id("job")
            self.queue.enqueue_call(
                func="app.workers.tasks.run_analysis",
                args=(analysis_id,),
                job_id=job_id,
                job_timeout=self.settings.job_timeout_seconds,
                result_ttl=86400,
                failure_ttl=604800,
            )
            return EnqueueResult(backend=self.name, job_id=job_id)
        except Exception as exc:  # noqa: BLE001 - stable external queue failure
            raise QueueUnavailable(
                "Redis/RQ could not accept the analysis job.",
                detail={"redis_url": _redact_redis(self.settings.redis_url), "error": str(exc)},
            ) from exc

    def health(self) -> dict[str, Any]:
        try:
            latency_started = __import__("time").perf_counter()
            self.connection.ping()
            latency_ms = (__import__("time").perf_counter() - latency_started) * 1000
            return {
                "status": "ok",
                "message": "Redis/RQ is reachable.",
                "backend": self.name,
                "authoritative": True,
                "queue": self.queue.name,
                "queued_jobs": self.queue.count,
                "redis_latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "message": "Redis/RQ is unavailable.",
                "backend": self.name,
                "authoritative": True,
                "error": str(exc),
            }

    def shutdown(self) -> None:
        try:
            self.connection.close()
        except Exception:  # noqa: BLE001
            pass


def _redact_redis(url: str) -> str:
    if "@" not in url:
        return url
    prefix, suffix = url.rsplit("@", 1)
    scheme = prefix.split("://", 1)[0]
    return f"{scheme}://***@{suffix}"


_queue: InlineQueue | RqQueue | None = None
_queue_lock = threading.Lock()


def get_queue(settings: Settings | None = None) -> InlineQueue | RqQueue:
    global _queue
    settings = settings or get_settings()
    with _queue_lock:
        expected = settings.queue_mode.value
        if _queue is not None and _queue.name != expected:
            _queue.shutdown()
            _queue = None
        if _queue is None:
            _queue = RqQueue(settings) if settings.queue_mode is QueueMode.RQ else InlineQueue(settings)
        return _queue


def shutdown_queue() -> None:
    global _queue
    with _queue_lock:
        if _queue is not None:
            _queue.shutdown()
        _queue = None
