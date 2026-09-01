"""Dataset manifest schema + geographic (scene-level) train/val/test split logic.

Implementation_Plan_v1.2.md: "group all tiles from the same source scene into one
split; prevent same-scene leakage; prevent overlapping/nearby tile leakage where
applicable." BigEarthNet patches are cut out of larger Sentinel-2 tiles; the tile name
(first part of the patch_id, e.g. ``S2A_MSIL2A_20170613T101031_N0205_R022_T33UUP``) is
the natural "scene" unit — patches from the same tile are geographically adjacent and
must never be split across train/val/test.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ml.sarfuseseg.config import DEFAULT_SEED


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    s2_patch_id: str
    s1_patch_name: str
    tile_id: str  # scene grouping key
    optical_path: str
    sar_path: str
    label_path: str | None
    split: str | None = None  # filled in by assign_geographic_splits
    label_provenance: str = "corine_clc2018_reference_map"
    label_confidence: float = 1.0  # lowered for coarse/weak labels (see filter_weak_labels)
    country: str | None = None
    season: str | None = None


MANIFEST_VERSION = "v1"


def write_manifest(entries: list[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": MANIFEST_VERSION, "count": len(entries), "entries": [asdict(e) for e in entries]}
    path.write_text(json.dumps(payload, indent=2))


def read_manifest(path: Path) -> list[ManifestEntry]:
    payload = json.loads(path.read_text())
    return [ManifestEntry(**e) for e in payload["entries"]]


def assign_geographic_splits(
    entries: list[ManifestEntry],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> list[ManifestEntry]:
    """Split by ``tile_id`` (scene), not by sample, so no scene straddles two splits.

    A simple deterministic shuffle-and-cut over the *tile* list. This does not attempt
    spatial buffering between neighbouring tiles (tiles across countries/dates in
    BigEarthNet are not adjacent in practice), but it does guarantee the primary
    leakage failure mode — the same tile's patches landing in both train and test —
    cannot happen.
    """
    tiles = sorted({e.tile_id for e in entries})
    rng = random.Random(seed)
    rng.shuffle(tiles)

    n = len(tiles)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    train_tiles = set(tiles[:n_train])
    val_tiles = set(tiles[n_train : n_train + n_val])
    test_tiles = set(tiles[n_train + n_val :])

    def split_of(tile_id: str) -> str:
        if tile_id in train_tiles:
            return "train"
        if tile_id in val_tiles:
            return "val"
        return "test"

    return [
        ManifestEntry(**{**asdict(e), "split": split_of(e.tile_id)})
        for e in entries
    ]


def manifest_summary(entries: list[ManifestEntry]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for e in entries:
        summary[e.split or "unassigned"] = summary.get(e.split or "unassigned", 0) + 1
    return summary
