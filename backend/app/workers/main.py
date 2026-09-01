"""RQ worker entry point: ``python -m app.workers.main``."""

from __future__ import annotations

import os
import socket
import threading

from app.core.config import QueueMode, get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import init_db
from app.workers.queue import RqQueue
from app.workers.tasks import _record_heartbeat

logger = get_logger("workers.main")


def _heartbeat_loop(stop: threading.Event, worker_name: str) -> None:
    while not stop.is_set():
        _record_heartbeat(worker_name, state="online", analysis_id=None)
        stop.wait(30.0)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.queue_mode is not QueueMode.RQ:
        raise SystemExit("The standalone worker requires SATQUERY_QUEUE_MODE=rq.")
    if settings.gpu_concurrency != 1:
        raise SystemExit("The MVP architecture requires SATQUERY_GPU_CONCURRENCY=1.")

    init_db()
    backend = RqQueue(settings)
    worker_name = os.environ.get("SATQUERY_WORKER_NAME") or f"satquery-{socket.gethostname()}-{os.getpid()}"
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(stop, worker_name),
        name="satquery-worker-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        from rq import Worker

        logger.info("starting RQ worker name=%s queue=%s", worker_name, backend.queue.name)
        Worker([backend.queue], connection=backend.connection, name=worker_name).work(with_scheduler=False)
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)
        backend.shutdown()


if __name__ == "__main__":
    main()
