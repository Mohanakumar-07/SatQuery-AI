"""FastAPI application factory for the SatQuery backend."""

from __future__ import annotations

import re
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.v1.router import router as v1_router
from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.core.ids import new_id
from app.core.logging import configure_logging, get_logger, get_request_id, set_request_id
from app.db.session import init_db
from app.schemas.health import RootResponse
from app.services.model_registry import get_registry
from app.workers.queue import get_queue, shutdown_queue

logger = get_logger("main")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,96}$")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_dirs()
        init_db(settings.database_url)
        app.state.settings = settings
        app.state.job_queue = get_queue(settings)
        app.state.started_monotonic = time.monotonic()
        get_registry(settings).load()
        logger.info(
            "application started env=%s queue=%s pipeline=%s",
            settings.env.value,
            settings.queue_mode.value,
            settings.pipeline_mode.value,
        )
        try:
            yield
        finally:
            shutdown_queue()
            logger.info("application stopped")

    app = FastAPI(
        title=settings.app_name,
        version=settings.version or __version__,
        description=(
            "Evidence-backed satellite-image analysis control plane. Users provide imagery "
            "and a question; the backend validates, interprets, routes, queues and serves "
            "structured evidence without exposing model selection."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Runtime capability and model registry."},
            {"name": "uploads", "description": "Secure raster ingestion and metadata."},
            {"name": "validation", "description": "Geospatial preflight checks."},
            {"name": "analyses", "description": "Asynchronous analysis lifecycle."},
            {"name": "artifacts", "description": "Web-compatible evidence artifacts."},
            {"name": "reports", "description": "Downloadable result reports."},
        ],
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Accept", "Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["Content-Disposition", "ETag", "X-Request-ID", "X-Process-Time-Ms"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "").strip()
        request_id = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else new_id("request")
        set_request_id(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            set_request_id("-")
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        set_request_id("-")
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_payload(getattr(request.state, "request_id", get_request_id())),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "location": [str(part) for part in item.get("loc", ())],
                "message": str(item.get("msg", "Invalid value.")),
                "type": str(item.get("type", "value_error")),
            }
            for item in exc.errors()
        ]
        error = AppError(
            "Request validation failed.",
            code=ErrorCode.INVALID_REQUEST,
            status_code=422,
            detail={"errors": errors},
        )
        return JSONResponse(
            status_code=422,
            content=error.to_payload(getattr(request.state, "request_id", get_request_id())),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error = AppError(
            str(exc.detail),
            code=ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_REQUEST,
            status_code=exc.status_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error.to_payload(getattr(request.state, "request_id", get_request_id())),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled request error path=%s", request.url.path)
        error = AppError(
            "The server could not complete the request.",
            code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
        )
        return JSONResponse(
            status_code=500,
            content=error.to_payload(getattr(request.state, "request_id", get_request_id())),
        )

    @app.get("/", response_model=RootResponse, include_in_schema=False)
    def root() -> RootResponse:
        return RootResponse(
            name=settings.app_name,
            version=settings.version,
            environment=settings.env.value,
            docs_url="/docs",
            openapi_url="/openapi.json",
            health_url=f"{settings.api_prefix}/health",
            api_prefix=settings.api_prefix,
        )

    app.include_router(v1_router, prefix=settings.api_prefix)
    return app


app = create_app()
