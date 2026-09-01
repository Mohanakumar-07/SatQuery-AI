"""Runtime configuration for the SatQuery backend.

Plain ``os.environ`` + a dataclass rather than ``pydantic-settings``: the web layer
must import cleanly on a bare machine before the geospatial and model stacks exist.
Values can also be supplied by a ``.env`` file at the repository root.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path


class QueueMode(str, Enum):
    """How analyses get from the API process into a worker (plan section 7.0)."""

    RQ = "rq"  # Redis + RQ, GPU worker in its own process
    INLINE = "inline"  # single in-process thread, development only


class PipelineMode(str, Enum):
    """Which pipeline the worker executes (plan section 5)."""

    UNATTACHED = "unattached"  # fail fast, PIPELINE_NOT_ATTACHED
    STUB = "stub"  # shaped, explicitly non-authoritative results
    PYTHON = "python"  # import settings.pipeline_callable


class Env(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"


def _repo_root() -> Path:
    # backend/app/core/config.py -> parents[3] == repository root
    return Path(__file__).resolve().parents[3]


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=value`` lines from a .env file into ``os.environ``.

    Real environment variables win unless ``override=True``. Lines that are blank or
    start with ``#`` are ignored, and surrounding quotes are stripped. This is a
    convenience for local development, not a full dotenv implementation.
    """
    env_path = path or _repo_root() / ".env"
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = _ENV_LINE.match(raw)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif "#" in value:  # strip trailing comment on unquoted values
            value = value.split("#", 1)[0].strip()
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return default if value is None or value == "" else value


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _optional_float(key: str) -> float | None:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _enum(enum_cls, key: str, default):
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    try:
        return enum_cls(raw)
    except ValueError:
        return default


def _path(key: str, default: str) -> Path:
    raw = Path(_str(key, default)).expanduser()
    return raw if raw.is_absolute() else _repo_root() / raw


def _resolve(path: Path) -> Path:
    return path.resolve()


def _database_url(value: str) -> str:
    """Normalise PostgreSQL for Psycopg 3 or resolve a repo-relative SQLite URL."""
    if value.startswith("postgres://"):
        return f"postgresql+psycopg://{value[len('postgres://') :]}"
    if value.startswith("postgresql://"):
        return f"postgresql+psycopg://{value[len('postgresql://') :]}"
    prefix = "sqlite:///"
    if not value.startswith(prefix):
        return value
    raw = value[len(prefix) :]
    if not raw or raw == ":memory:" or raw.startswith("/") or Path(raw).is_absolute():
        return value
    absolute = (_repo_root() / raw).resolve().as_posix()
    return f"{prefix}{absolute}"


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the backend environment."""

    # ---- service ----
    app_name: str
    version: str
    env: Env
    api_prefix: str
    log_level: str
    cors_origins: tuple[str, ...]

    # ---- storage ----
    repo_root: Path
    artifacts_dir: Path
    state_dir: Path
    database_url: str

    # ---- uploads / validation limits ----
    allowed_extensions: frozenset[str]
    max_upload_bytes: int
    max_decompressed_bytes: int
    max_raster_pixels: int
    max_files_per_analysis: int
    min_overlap_percent: float
    max_residual_offset_pixels: float | None

    # ---- async execution ----
    queue_mode: QueueMode
    redis_url: str
    job_timeout_seconds: int
    gpu_concurrency: int

    # ---- pipeline seam ----
    pipeline_mode: PipelineMode
    pipeline_callable: str

    # ---- registry / calibration ----
    model_registry_path: Path
    confidence_policy_path: Path

    # ---- reports / URLs ----
    public_base_url: str
    report_base_url: str

    # ---- derived (see properties below) ----
    _env_present: bool = field(default=False, repr=False)

    @property
    def is_test(self) -> bool:
        return self.env is Env.TEST

    @property
    def is_production(self) -> bool:
        return self.env is Env.PRODUCTION

    @property
    def uploads_dir(self) -> Path:
        return self.artifacts_dir / "uploads"

    @property
    def masks_dir(self) -> Path:
        return self.artifacts_dir / "masks"

    @property
    def geojson_dir(self) -> Path:
        return self.artifacts_dir / "geojson"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def scenes_dir(self) -> Path:
        return self.artifacts_dir / "scenes"

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def sqlite_path(self) -> Path | None:
        """Filesystem path of the SQLite database, when SQLite is in use."""
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            return None
        raw = self.database_url[len(prefix) :]
        if not raw or raw == ":memory:":
            return None
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self.repo_root / candidate

    def artifact_url(self, relative_path: str) -> str:
        """Absolute URL for a stored artifact, when a public base URL is configured."""
        base = (self.report_base_url or self.public_base_url).rstrip("/")
        rel = relative_path.lstrip("/")
        return f"{base}/{rel}" if base else f"/{rel}"

    def ensure_dirs(self) -> None:
        for directory in (
            self.artifacts_dir,
            self.uploads_dir,
            self.masks_dir,
            self.geojson_dir,
            self.reports_dir,
            self.scenes_dir,
            self.checkpoints_dir,
            self.state_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        sqlite = self.sqlite_path
        if sqlite is not None:
            sqlite.parent.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, object]:
        """Environment-safe view for /health and logs."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "env": self.env.value,
            "api_prefix": self.api_prefix,
            "artifacts_dir": str(self.artifacts_dir),
            "database_url": _redact_database_url(self.database_url),
            "queue_mode": self.queue_mode.value,
            "pipeline_mode": self.pipeline_mode.value,
            "pipeline_callable": self.pipeline_callable if self.pipeline_mode is PipelineMode.PYTHON else None,
            "gpu_concurrency": self.gpu_concurrency,
            "max_upload_bytes": self.max_upload_bytes,
            "max_files_per_analysis": self.max_files_per_analysis,
            "min_overlap_percent": self.min_overlap_percent,
            "max_residual_offset_pixels": self.max_residual_offset_pixels,
            "cors_origins": list(self.cors_origins),
        }


def _redact_database_url(url: str) -> str:
    return re.sub(r"://([^:/?]+):[^@]+@", r"://\1:***@", url)


def build_settings() -> Settings:
    """Read ``.env`` (if any) and build an immutable settings snapshot."""
    env_present = load_dotenv()
    origins = _str(
        "SATQUERY_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,http://localhost:5173,http://127.0.0.1:5173",
    )
    cors = tuple(o.strip() for o in origins.split(",") if o.strip())
    extensions = _str("SATQUERY_ALLOWED_EXTENSIONS", "tif,tiff,gtif,png,jpg,jpeg")
    database_url = (
        os.environ.get("SATQUERY_DATABASE_URL")
        or os.environ.get("DATABASE_URL_POOLED")
        or os.environ.get("DATABASE_URL")
        or "sqlite:///var/satquery.db"
    )

    settings = Settings(
        app_name=_str("SATQUERY_APP_NAME", "SatQuery AI API"),
        version=_str("SATQUERY_VERSION", "0.1.0"),
        env=_enum(Env, "SATQUERY_ENV", Env.DEVELOPMENT),
        api_prefix=_str("SATQUERY_API_PREFIX", "/api/v1").rstrip("/") or "/api/v1",
        log_level=_str("SATQUERY_LOG_LEVEL", "INFO").upper(),
        cors_origins=cors,
        repo_root=_repo_root(),
        artifacts_dir=_resolve(_path("SATQUERY_ARTIFACTS_DIR", "artifacts")),
        state_dir=_resolve(_path("SATQUERY_STATE_DIR", "var")),
        database_url=_database_url(database_url),
        allowed_extensions=frozenset(e.strip().lower().lstrip(".") for e in extensions.split(",") if e.strip()),
        max_upload_bytes=_int("SATQUERY_MAX_UPLOAD_BYTES", 500 * 1024 * 1024),
        max_decompressed_bytes=_int("SATQUERY_MAX_DECOMPRESSED_BYTES", 2 * 1024 * 1024 * 1024),
        max_raster_pixels=_int("SATQUERY_MAX_RASTER_PIXELS", 16384 * 16384),
        max_files_per_analysis=_int("SATQUERY_MAX_FILES_PER_ANALYSIS", 2),
        min_overlap_percent=_float("SATQUERY_MIN_OVERLAP_PERCENT", 20.0),
        max_residual_offset_pixels=_optional_float("SATQUERY_MAX_RESIDUAL_OFFSET_PIXELS"),
        queue_mode=_enum(QueueMode, "SATQUERY_QUEUE_MODE", QueueMode.INLINE),
        redis_url=_str("SATQUERY_REDIS_URL", "redis://localhost:6379/0"),
        job_timeout_seconds=_int("SATQUERY_JOB_TIMEOUT_SECONDS", 3600),
        gpu_concurrency=max(1, _int("SATQUERY_GPU_CONCURRENCY", 1)),
        pipeline_mode=_enum(PipelineMode, "SATQUERY_PIPELINE_MODE", PipelineMode.UNATTACHED),
        pipeline_callable=_str("SATQUERY_PIPELINE_CALLABLE", "app.pipeline.runner.execute"),
        model_registry_path=_resolve(_path("SATQUERY_MODEL_REGISTRY_PATH", "backend/app/config/model_registry.json")),
        confidence_policy_path=_resolve(
            _path("SATQUERY_CONFIDENCE_POLICY_PATH", "backend/app/config/confidence_policies.json")
        ),
        public_base_url=_str("SATQUERY_PUBLIC_BASE_URL", ""),
        report_base_url=_str("SATQUERY_REPORT_BASE_URL", ""),
        _env_present=bool(env_present),
    )
    settings.ensure_dirs()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return build_settings()


def clear_settings_cache() -> None:
    """Drop the cached settings (used by tests that patch the environment)."""
    get_settings.cache_clear()
