"""
SARFuseSegSegmenter -- Encapsulated direct and versioned tiled inference for SAR-FuseSeg V3.
Dynamically loads normalization parameters from normalization_stats.json as the single source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .models.sar_fuse_seg import SARFuseSeg


class SARFuseSegSegmenter:
    """
    Manages SAR-FuseSeg V3 model inference, dynamic normalization,
    frozen 120x120 direct inference, and separately versioned tiled inference.
    """

    DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "sar_fuse_v3" / "best.pt"
    DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.DEFAULT_CHECKPOINT

        self.normalization_stats = self._load_normalization_stats()
        self.label_mapping = self._load_label_mapping()
        self.model = self._load_model()

    def _load_normalization_stats(self) -> Dict[str, Any]:
        stats_file = self.config_dir / "normalization_stats.json"
        if not stats_file.exists():
            raise FileNotFoundError(f"Normalization statistics file not found: {stats_file}")
        with open(stats_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_label_mapping(self) -> Dict[str, Any]:
        mapping_file = self.config_dir / "label_mapping.json"
        if not mapping_file.exists():
            return {0: "built-up", 1: "water", 2: "vegetation"}
        with open(mapping_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self) -> SARFuseSeg:
        model = SARFuseSeg(optical_channels=6, sar_channels=2, num_classes=3)
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            if len(missing) > 0 or len(unexpected) > 0:
                raise RuntimeError(
                    f"Strict checkpoint loading failed. Missing: {missing}, Unexpected: {unexpected}"
                )
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        model.to(self.device)
        model.eval()
        return model

    def normalize_optical(self, optical: np.ndarray) -> np.ndarray:
        """
        Normalize 6-channel optical array (shape: [6, H, W] or [H, W, 6]).
        Dynamic means and stds loaded from normalization_stats.json.
        """
        mean = np.array(self.normalization_stats["optical"]["mean"], dtype=np.float32)
        std = np.array(self.normalization_stats["optical"]["std"], dtype=np.float32)

        if optical.shape[0] == 6:
            mean = mean[:, None, None]
            std = std[:, None, None]
        elif optical.shape[-1] == 6:
            mean = mean[None, None, :]
            std = std[None, None, :]
        else:
            raise ValueError(f"Expected 6 optical channels, got shape {optical.shape}")

        return (optical.astype(np.float32) - mean) / std

    def normalize_sar(self, sar: np.ndarray) -> np.ndarray:
        """
        Normalize 2-channel SAR array (shape: [2, H, W] or [H, W, 2]).
        Dynamic means and stds loaded from normalization_stats.json.
        """
        mean = np.array(self.normalization_stats["sar"]["mean"], dtype=np.float32)
        std = np.array(self.normalization_stats["sar"]["std"], dtype=np.float32)

        if sar.shape[0] == 2:
            mean = mean[:, None, None]
            std = std[:, None, None]
        elif sar.shape[-1] == 2:
            mean = mean[None, None, :]
            std = std[None, None, :]
        else:
            raise ValueError(f"Expected 2 SAR channels, got shape {sar.shape}")

        return (sar.astype(np.float32) - mean) / std

    @torch.no_grad()
    def infer_direct_120x120(
        self, optical: torch.Tensor, sar: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Frozen V3 direct-inference protocol strictly for 120x120 patches.
        optical: [1, 6, 120, 120] (or [6, 120, 120])
        sar: [1, 2, 120, 120] (or [2, 120, 120])
        """
        if optical.ndim == 3:
            optical = optical.unsqueeze(0)
        if sar.ndim == 3:
            sar = sar.unsqueeze(0)

        _, opt_c, opt_h, opt_w = optical.shape
        _, sar_c, sar_h, sar_w = sar.shape

        if (opt_c, opt_h, opt_w) != (6, 120, 120):
            raise ValueError(
                f"Direct inference strictly requires 120x120 patch with 6 optical bands. Got: {optical.shape}"
            )
        if (sar_c, sar_h, sar_w) != (2, 120, 120):
            raise ValueError(
                f"Direct inference strictly requires 120x120 patch with 2 SAR bands. Got: {sar.shape}"
            )

        optical = optical.to(self.device)
        sar = sar.to(self.device)

        logits = self.model(optical, sar)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # [3, 120, 120]
        class_map = np.argmax(probs, axis=0).astype(np.uint8)     # [120, 120]

        metadata = {
            "model": "SAR-FuseSeg-V3",
            "protocol": "sar_fuse_v3_direct",
            "input_resolution": [120, 120],
            "optical_channels": 6,
            "sar_channels": 2,
            "num_classes": 3,
            "device": str(self.device),
        }
        return class_map, probs, metadata

    @torch.no_grad()
    def infer_tiled_v1(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        tile_size: int = 120,
        overlap: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Separately versioned tiled inference protocol ('sar_fuse_v3_tiled_v1')
        for rasters larger than 120x120, with overlapping windows and linear/cosine blending.
        """
        if optical.ndim == 3:
            optical = optical.unsqueeze(0)
        if sar.ndim == 3:
            sar = sar.unsqueeze(0)

        _, opt_c, h, w = optical.shape
        _, sar_c, sar_h, sar_w = sar.shape

        if opt_c != 6 or sar_c != 2:
            raise ValueError(f"Expected 6 optical, 2 SAR channels. Got optical: {opt_c}, SAR: {sar_c}")
        if (h, w) != (sar_h, sar_w):
            raise ValueError(f"Spatial dimension mismatch between optical ({h}, {w}) and SAR ({sar_h}, {sar_w})")

        stride = tile_size - overlap
        prob_accum = np.zeros((3, h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)

        # 2D Hann / cosine window for smooth blending across overlaps
        win_1d = np.hanning(tile_size).astype(np.float32)
        win_1d = np.maximum(win_1d, 0.05)
        window = np.outer(win_1d, win_1d)

        y_steps = list(range(0, max(1, h - tile_size + 1), stride))
        if len(y_steps) == 0 or y_steps[-1] + tile_size < h:
            y_steps.append(max(0, h - tile_size))

        x_steps = list(range(0, max(1, w - tile_size + 1), stride))
        if len(x_steps) == 0 or x_steps[-1] + tile_size < w:
            x_steps.append(max(0, w - tile_size))

        for y in y_steps:
            for x in x_steps:
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                actual_h = y_end - y
                actual_w = x_end - x

                opt_tile = optical[:, :, y:y_end, x:x_end]
                sar_tile = sar[:, :, y:y_end, x:x_end]

                # Pad to tile_size if boundary tile is smaller
                pad_h = tile_size - actual_h
                pad_w = tile_size - actual_w
                if pad_h > 0 or pad_w > 0:
                    opt_tile = F.pad(opt_tile, (0, pad_w, 0, pad_h), mode="reflect")
                    sar_tile = F.pad(sar_tile, (0, pad_w, 0, pad_h), mode="reflect")

                opt_tile = opt_tile.to(self.device)
                sar_tile = sar_tile.to(self.device)

                logits = self.model(opt_tile, sar_tile)
                tile_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # [3, tile_size, tile_size]

                w_tile = window[:actual_h, :actual_w]
                prob_accum[:, y:y_end, x:x_end] += tile_probs[:, :actual_h, :actual_w] * w_tile[None, :, :]
                weight_accum[y:y_end, x:x_end] += w_tile

        # Normalize by accumulated weights
        weight_accum = np.maximum(weight_accum, 1e-6)
        prob_accum = prob_accum / weight_accum[None, :, :]
        # Re-normalize along class axis to ensure exact probability simplex
        prob_accum = prob_accum / np.sum(prob_accum, axis=0, keepdims=True)

        class_map = np.argmax(prob_accum, axis=0).astype(np.uint8)

        metadata = {
            "model": "SAR-FuseSeg-V3",
            "protocol": "sar_fuse_v3_tiled_v1",
            "tile_size": tile_size,
            "overlap": overlap,
            "stride": stride,
            "blending": "hann_window",
            "input_resolution": [h, w],
            "optical_channels": 6,
            "sar_channels": 2,
            "num_classes": 3,
            "device": str(self.device),
        }
        return class_map, prob_accum, metadata

