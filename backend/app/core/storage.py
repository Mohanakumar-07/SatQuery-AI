"""Artifact storage: where bytes live on disk and how they are resolved back.

Layout (all under ``settings.artifacts_dir``)::

    uploads/<upload_id>/<stored-name>       original client files, never modified
    scenes/<analysis_id>/                   canonical scene bundles (see preprocessing)
    masks/<analysis_id>/                    model masks and PNG overlays
    geojson/<analysis_id>/                  region vectors
    reports/<analysis_id>/                  generated report files
    checkpoints/                            model weights (ML-owned, never written by the API)

Only containment-checked paths leave this class: an artifact is always addressed as
``<kind>/<scope_id>/<name>`` and the resolved path is verified to stay inside the
artifacts root, so a crafted name can never escape to the filesystem.
"""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.errors import BadRequest, NotFound
from app.core.ids import safe_filename

#: Kinds the API is allowed to write. Anything else is rejected.
WRITABLE_KINDS = ("uploads", "scenes", "masks", "geojson", "reports")

_EXT_MEDIA_TYPES = {
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".gtif": "image/tiff",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".geojson": "application/geo+json",
    ".json": "application/json",
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".npy": "application/octet-stream",
    ".mask": "application/octet-stream",
}

_INLINE_SAFE_TYPES = frozenset({"image/png", "image/jpeg", "image/tiff", "text/plain", "text/csv"})


@dataclass(frozen=True)
class StoredObject:
    """A file the store has accepted."""

    absolute_path: Path
    relative_path: str
    size_bytes: int
    sha256: str

    @property
    def media_type(self) -> str:
        return media_type_for(self.absolute_path.name)


class PathTraversal(ValueError):
    """Raised when a resolved path escapes the artifacts root."""


def media_type_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in _EXT_MEDIA_TYPES:
        return _EXT_MEDIA_TYPES[ext]
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def is_inline_safe(media_type: str) -> bool:
    """Whether a browser may render this type inline rather than download it."""
    return media_type in _INLINE_SAFE_TYPES or media_type.startswith("image/") or media_type.startswith("text/")


class ArtifactStore:
    """Filesystem artifact store rooted at ``settings.artifacts_dir``."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._settings = self.settings
        self.root = self.settings.artifacts_dir.resolve()

    # ------------------------------------------------------------------ paths
    def kind_root(self, kind: str) -> Path:
        if kind not in WRITABLE_KINDS:
            raise BadRequest(
                f"Unknown artifact kind '{kind}'.",
                detail={"allowed": list(WRITABLE_KINDS)},
            )
        path = (self.root / kind).resolve()
        if not path.is_relative_to(self.root):
            raise PathTraversal("Artifact kind escaped the storage root.")
        return path

    def scope_dir(self, kind: str, scope_id: str) -> Path:
        """Directory for one upload or analysis, created on demand."""
        safe_scope = safe_filename(scope_id, fallback=kind)
        path = self.kind_root(kind) / safe_scope
        if not path.resolve().is_relative_to(self.root):
            raise PathTraversal("Artifact scope escaped the storage root.")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve(self, kind: str, scope_id: str, name: str) -> Path:
        """Containment-checked path for a stored artifact."""
        base = self.scope_dir(kind, scope_id)
        safe_name = safe_filename(name, fallback="artifact.bin")
        candidate = (base / safe_name).resolve()
        if not candidate.is_relative_to(base):
            raise PathTraversal("Artifact path escaped its scope directory.")
        return candidate

    def relative(self, path: Path) -> str:
        """POSIX-style path relative to the artifacts root (what the DB stores)."""
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError as exc:  # not under root
            raise BadRequest("Artifact path is outside the storage root.") from exc

    def from_relative(self, relative_path: str) -> Path:
        """Containment-checked resolution of a DB-stored relative path."""
        if not relative_path or relative_path.startswith(("/", "\\")) or "\x00" in relative_path:
            raise BadRequest("Stored artifact path is not usable.")
        candidate = (self.root / relative_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise PathTraversal("Stored artifact path escaped the storage root.")
        return candidate

    # ---------------------------------------------------------------- writing
    def write_bytes(self, kind: str, scope_id: str, name: str, data: bytes) -> StoredObject:
        path = self.resolve(kind, scope_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return self.stat_object(path)

    def write_stream(self, kind: str, scope_id: str, name: str, source, *, max_bytes: int | None = None) -> StoredObject:
        """Stream a file-like object into storage in fixed chunks.

        The size cap is enforced while copying, so an oversized upload is abandoned
        mid-transfer rather than filling the disk first.
        """
        path = self.resolve(kind, scope_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.part")
        digest = hashlib.sha256()
        total = 0
        try:
            with tmp.open("wb") as handle:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise BadRequest(
                            f"Upload exceeds the configured limit of {max_bytes} bytes.",
                            detail={"max_bytes": max_bytes, "received_before_abort": total},
                        )
                    digest.update(chunk)
                    handle.write(chunk)
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return StoredObject(
            absolute_path=path,
            relative_path=self.relative(path),
            size_bytes=total,
            sha256=digest.hexdigest(),
        )

    def read_bytes(self, kind: str, scope_id: str, name: str) -> bytes:
        path = self.resolve(kind, scope_id, name)
        if not path.is_file():
            raise ArtifactNotFoundForScope(kind, scope_id, name)
        return path.read_bytes()

    def delete_scope(self, kind: str, scope_id: str) -> bool:
        base = self.kind_root(kind) / safe_filename(scope_id, fallback=kind)
        resolved = base.resolve()
        if not resolved.is_relative_to(self.root) or not resolved.exists():
            return False
        shutil.rmtree(resolved, ignore_errors=True)
        return True

    # ---------------------------------------------------------------- reading
    def stat_object(self, path: Path) -> StoredObject:
        resolved = path.resolve()
        stat = resolved.stat()
        return StoredObject(
            absolute_path=resolved,
            relative_path=self.relative(resolved),
            size_bytes=stat.st_size,
            sha256=sha256_file(resolved),
        )

    def free_disk_bytes(self) -> int | None:
        try:
            return shutil.disk_usage(self.root).free
        except OSError:
            return None

    def describe(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "writable_kinds": list(WRITABLE_KINDS),
            "free_disk_bytes": self.free_disk_bytes(),
        }


class ArtifactNotFoundForScope(NotFound):
    def __init__(self, kind: str, scope_id: str, name: str) -> None:
        super().__init__(
            "The requested artifact file is missing from storage.",
            detail={"kind": kind, "scope_id": scope_id, "name": name},
        )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


_default_store: ArtifactStore | None = None


def get_store(settings: Settings | None = None) -> ArtifactStore:
    """Process-wide store for the current settings."""
    global _default_store
    if _default_store is None or (settings is not None and settings is not _default_store.settings):
        _default_store = ArtifactStore(settings)
    return _default_store


def reset_store_cache() -> None:
    global _default_store
    _default_store = None
