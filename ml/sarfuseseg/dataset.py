"""Optical-only, SAR-only and paired optical+SAR PyTorch datasets for SAR-FuseSeg.

BigEarthNet v2.0 patches are natively 120x120 px at 10 m GSD (1.2 km x 1.2 km tiles) —
smaller than the 256x256 tile size this project standardizes on for the model input
contract. For this first pass we reflect-pad each patch up to 256x256 (documented
here, not hidden) rather than mosaicking neighbouring patches; that is a reasonable
follow-up once the baseline loader/training loop is proven.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from ml.sarfuseseg.class_mapping import ClassMapping, remap_label_array
from ml.sarfuseseg.config import IGNORE_INDEX, OPTICAL_BANDS, SAR_BANDS, TILE_SIZE
from ml.sarfuseseg.manifest import ManifestEntry
from ml.sarfuseseg.preprocessing import SARFuseSegPreprocessor


def _read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def _pad_to_tile(arr: np.ndarray, tile_size: int = TILE_SIZE, fill: float | None = None) -> np.ndarray:
    """Reflect-pad (or constant-pad labels) a (..., H, W) array up to tile_size x tile_size."""
    *_, h, w = arr.shape
    pad_h, pad_w = max(0, tile_size - h), max(0, tile_size - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)]
    mode = "constant" if fill is not None else "reflect"
    kwargs = {"constant_values": fill} if fill is not None else {}
    return np.pad(arr, pad_width, mode=mode, **kwargs)


def _load_optical_bands(patch_dir: Path) -> dict[str, np.ndarray]:
    return {band: _read_band(patch_dir / f"{patch_dir.name}_{band}.tif") for band in OPTICAL_BANDS}


def _load_sar_bands(patch_dir: Path) -> dict[str, np.ndarray]:
    return {band: _read_band(patch_dir / f"{patch_dir.name}_{band}.tif") for band in SAR_BANDS}


def _load_label(label_path: Path | None, shape: tuple[int, int], mapping: ClassMapping, id_to_name: dict[int, str]) -> np.ndarray:
    if label_path is None:
        return np.full(shape, IGNORE_INDEX, dtype=np.uint8)
    clc_ids = _read_band(label_path)
    return remap_label_array(clc_ids, id_to_name, mapping)


@dataclass
class Sample:
    """Shape reference only — ``__getitem__`` returns a plain dict (see below) so the
    default ``DataLoader`` collate function can stack batches without a custom collate_fn.
    """

    optical: torch.Tensor  # (C,H,W)
    sar: torch.Tensor  # (2,H,W)
    label: torch.Tensor  # (H,W) long
    valid_mask: torch.Tensor  # (H,W) bool
    sample_id: str


class _BaseSARFuseSegDataset(Dataset):
    def __init__(
        self,
        entries: list[ManifestEntry],
        mapping: ClassMapping,
        clc_id_to_name: dict[int, str],
        preprocessor: SARFuseSegPreprocessor | None = None,
        blank_optical: bool = False,
        blank_sar: bool = False,
        mismatch_pairs: bool = False,
    ):
        self.entries = entries
        self.mapping = mapping
        self.clc_id_to_name = clc_id_to_name
        self.preprocessor = preprocessor or SARFuseSegPreprocessor()
        self.blank_optical = blank_optical
        self.blank_sar = blank_sar
        self.mismatch_pairs = mismatch_pairs  # Experiment 6: deliberately shuffled SAR

    def __len__(self) -> int:
        return len(self.entries)

    def _sar_entry_for(self, index: int) -> ManifestEntry:
        if not self.mismatch_pairs:
            return self.entries[index]
        # Experiment 6: pair each optical sample with a SAR patch from a *different*
        # sample (fixed offset so it is deterministic/reproducible), to verify the
        # model/metrics correctly degrade on a mismatched, non-corresponding pair.
        return self.entries[(index + 1) % len(self.entries)]

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        sar_entry = self._sar_entry_for(index)

        optical_bands = _load_optical_bands(Path(entry.optical_path))
        sar_bands = _load_sar_bands(Path(sar_entry.sar_path))

        prepared = self.preprocessor.prepare(optical_bands, sar_bands)
        optical_tensor = prepared.optical.tensor
        sar_tensor = prepared.sar.tensor
        valid_mask = prepared.valid_mask

        if self.blank_optical:
            optical_tensor = np.zeros_like(optical_tensor)
        if self.blank_sar:
            sar_tensor = np.zeros_like(sar_tensor)

        label = _load_label(
            Path(entry.label_path) if entry.label_path else None,
            optical_tensor.shape[-2:],
            self.mapping,
            self.clc_id_to_name,
        )
        label[~valid_mask] = IGNORE_INDEX

        optical_tensor = _pad_to_tile(optical_tensor)
        sar_tensor = _pad_to_tile(sar_tensor)
        label = _pad_to_tile(label, fill=IGNORE_INDEX)
        valid_mask = _pad_to_tile(valid_mask.astype(np.uint8), fill=0).astype(bool)

        return {
            "optical": torch.from_numpy(optical_tensor.copy()).float(),
            "sar": torch.from_numpy(sar_tensor.copy()).float(),
            "label": torch.from_numpy(label.copy()).long(),
            "valid_mask": torch.from_numpy(valid_mask.copy()).bool(),
            "sample_id": entry.sample_id,
        }


class PairedOpticalSARDataset(_BaseSARFuseSegDataset):
    """Experiment 3 (fusion) and Experiment 6 (mismatch, via mismatch_pairs=True)."""


class OpticalOnlyDataset(_BaseSARFuseSegDataset):
    """Experiment 1: optical-only baseline (SAR branch fed zeros)."""

    def __init__(self, *args, **kwargs):
        kwargs["blank_sar"] = True
        super().__init__(*args, **kwargs)


class SAROnlyDataset(_BaseSARFuseSegDataset):
    """Experiment 2: SAR-only baseline (optical branch fed zeros)."""

    def __init__(self, *args, **kwargs):
        kwargs["blank_optical"] = True
        super().__init__(*args, **kwargs)
