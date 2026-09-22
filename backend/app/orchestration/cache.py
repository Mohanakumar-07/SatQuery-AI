"""Analysis reuse and cache identity according to plan sections 20 & 21.

Constructs deterministic SHA-256 cache keys to guarantee:
- same inputs + same model + same checkpoint + same params => reuse existing analysis
- any change in input, checkpoint, threshold or version => cache miss, rerun specialist
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List, Optional


def compute_analysis_cache_key(
    *,
    input_hashes: list[str],
    specialist: str,
    specialist_version: str,
    checkpoint_sha256: Optional[str] = None,
    preprocessing_version: Optional[str] = None,
    threshold: Optional[float] = None,
    params: Optional[dict[str, Any]] = None,
) -> str:
    """Calculates an immutable SHA-256 cache identity."""
    hasher = hashlib.sha256()

    # Sort input hashes to be order-deterministic where appropriate
    for h in input_hashes:
        hasher.update(h.encode("utf-8"))

    hasher.update(specialist.strip().lower().encode("utf-8"))
    hasher.update(specialist_version.strip().encode("utf-8"))

    if checkpoint_sha256:
        hasher.update(checkpoint_sha256.strip().encode("utf-8"))
    if preprocessing_version:
        hasher.update(preprocessing_version.strip().encode("utf-8"))
    if threshold is not None:
        hasher.update(f"{threshold:.4f}".encode("utf-8"))
    if params:
        param_str = json.dumps(params, sort_keys=True)
        hasher.update(param_str.encode("utf-8"))

    return hasher.hexdigest()
