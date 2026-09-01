"""Load and apply the versioned SAR-FuseSeg class mapping (see manifests/class_mapping_v1.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.sarfuseseg.config import CLASS_NAMES, IGNORE_INDEX, MANIFEST_DIR

DEFAULT_MAPPING_PATH = MANIFEST_DIR / "class_mapping_v1.json"


@dataclass(frozen=True)
class ClassMapping:
    version: str
    ignore_index: int
    class_names: list[str]
    label_to_class: dict[str, str]  # source CLC label -> target class name

    def class_index(self, class_name: str) -> int:
        return self.class_names.index(class_name)

    def label_to_index(self, source_label: str) -> int:
        target = self.label_to_class.get(source_label)
        return self.ignore_index if target is None else self.class_index(target)


def load_class_mapping(path: Path = DEFAULT_MAPPING_PATH) -> ClassMapping:
    payload = json.loads(path.read_text())
    return ClassMapping(
        version=payload["version"],
        ignore_index=payload["ignore_index"],
        class_names=payload["target_classes"],
        label_to_class=payload["mapping"],
    )


def remap_label_array(clc_label_ids: np.ndarray, id_to_name: dict[int, str], mapping: ClassMapping) -> np.ndarray:
    """Map a raster of CLC numeric IDs to the 4-class SAR-FuseSeg label array.

    Any ID missing from ``id_to_name``/``mapping`` becomes ``IGNORE_INDEX`` — coarse or
    unresolvable labels are dropped, never guessed (plan: "filtering of invalid labels").
    """
    out = np.full(clc_label_ids.shape, IGNORE_INDEX, dtype=np.uint8)
    for clc_id, name in id_to_name.items():
        target_idx = mapping.label_to_index(name)
        out[clc_label_ids == clc_id] = target_idx
    return out
