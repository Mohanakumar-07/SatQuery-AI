"""Identifier generation and safe file naming.

IDs are time-sortable and prefixed (``upload-``, ``analysis-``, ``artifact-``) so
they read like the plan's examples while staying globally unique across workers.
"""

from __future__ import annotations

import os
import re
import time
import uuid

# Crockford-ish alphabet without characters that are easy to mistype in a URL.
_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_MULTI_DOT = re.compile(r"\.{2,}")


def _b32(value: int, width: int) -> str:
    """Encode a non-negative integer in the 32-character ID alphabet, big-endian."""
    chars = ["0"] * width
    for index in range(width - 1, -1, -1):
        chars[index] = _ALPHABET[value & 31]
        value >>= 5
    return "".join(chars)


def new_id(prefix: str, *, when: float | None = None) -> str:
    """Return a sortable identifier such as ``analysis-4kq2m9xtrbh1zq0dfgkm``.

    The first 9 characters encode the creation time in milliseconds, so a list of
    IDs sorts chronologically without a database round trip; the remaining 7 are
    random. 45 bits of timestamp and 35 bits of entropy.
    """
    timestamp_ms = int((time.time() if when is None else when) * 1000)
    entropy = int.from_bytes(os.urandom(8), "big")
    return f"{prefix}-{_b32(timestamp_ms, 9)}{_b32(entropy, 7)}"


def short_hash(data: bytes, length: int = 16) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, data.hex()[: length * 8]).hex[:length]


def safe_filename(name: str, *, fallback: str = "upload.bin", max_length: int = 120) -> str:
    """Reduce a client-supplied filename to a single safe path component.

    Strips directory parts (including Windows drives and backslashes), rejects
    traversal and dot-only names, collapses repeated dots, and preserves a visible
    extension when one is present.
    """
    cleaned = (name or "").replace("\\", "/").split("/")[-1].strip()
    cleaned = _UNSAFE.sub("_", cleaned)
    cleaned = _MULTI_DOT.sub(".", cleaned).strip("._-")
    if not cleaned or cleaned in {".", ".."}:
        return fallback
    stem, dot, ext = cleaned.rpartition(".")
    if dot and len(ext) <= 8 and stem:
        stem = stem[: max_length - len(ext) - 1]
        cleaned = f"{stem}.{ext.lower()}"
    else:
        cleaned = cleaned[:max_length]
    return cleaned or fallback


def extension_of(name: str) -> str:
    """Lowercase extension without the dot, or an empty string."""
    cleaned = (name or "").replace("\\", "/").split("/")[-1]
    if "." not in cleaned or cleaned.startswith("."):
        return ""
    return cleaned.rsplit(".", 1)[-1].lower()


def sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
