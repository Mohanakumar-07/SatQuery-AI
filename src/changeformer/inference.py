"""Inference helpers for the bundled ChangeFormer V6 LEVIR checkpoint."""
"""
inference.py -- Production inference engine for ChangeFormer V6 (OSCD Champion).

Supports arbitrary image dimensions via seamless sliding-window tiled inference,
reflection padding for undersized scenes, frozen z-score normalization,
calibrated decision threshold (tau=0.25), and 4-panel visual verification.
inference.py -- Compatibility redirect to detector.py
"""

from __future__ import annotations
from .detector import ChangeDetector, DEFAULT_CHECKPOINT, FROZEN_THRESHOLD

import json
from pathlib import Path
from typing import Union, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .ChangeFormer import ChangeFormerV6

# Frozen OSCD Champion Defaults
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "oscd" / "v3_1_controlled_sampler" / "checkpoints" / "best.pt"
)
FROZEN_MEAN = [0.151, 0.144, 0.146]
FROZEN_STD  = [0.084, 0.060, 0.047]
FROZEN_THRESHOLD = 0.25

def load_model(checkpoint_path: Path, device: torch.device) -> ChangeFormerV6:
    """Load the upstream V6 model from a ``best_ckpt.pt`` checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_G_state_dict"]
    model = ChangeFormerV6(embed_dim=256)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

class ChangeDetector:
    """
    Production inference wrapper for ChangeFormer V6.
    
    Usage:
        detector = ChangeDetector()
        results = detector.predict("path/to/t1.png", "path/to/t2.png")
        detector.save_visualization(results, "output_comparison.png")
    """

def image_to_tensor(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    """Load an RGB image using the normalization used by ChangeFormer training."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
        threshold: float = FROZEN_THRESHOLD,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mean: list[float] = FROZEN_MEAN,
        std: list[float] = FROZEN_STD,
        tile_size: int = 256,
        tile_stride: int = 128,
    ):
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        self.device = dev
        self.threshold = threshold
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.mean = mean
        self.std = std

    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)
        self.normalise = transforms.Normalize(mean=mean, std=std)
        self.model = self._load_model(Path(checkpoint_path))

        # 2D Hann blending window for smooth sliding-window aggregation
        hann_1d = np.hanning(self.tile_size)
        self.window = np.outer(hann_1d, hann_1d).astype(np.float32)
        self.window = np.maximum(self.window, 1e-4)

@torch.inference_mode()
def predict_change_mask(
    model: ChangeFormerV6,
    image_a: Path,
    image_b: Path,
    image_size: int,
    device: torch.device,
) -> Image.Image:
    """Return a binary 8-bit change mask for a before/after image pair."""
    logits = model(
        image_to_tensor(image_a, image_size, device),
        image_to_tensor(image_b, image_size, device),
    )[-1]
    mask = torch.argmax(logits, dim=1)[0].to(torch.uint8).cpu().numpy() * 255
    return Image.fromarray(mask, mode="L")
    def _load_model(self, ckpt_path: Path) -> ChangeFormerV6:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"ChangeFormer checkpoint not found at: {ckpt_path}")

        model = ChangeFormerV6(embed_dim=256)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Support both fine-tuned OSCD keys and upstream LEVIR keys
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model_G_state_dict" in ckpt:
            state_dict = ckpt["model_G_state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_image(self, img_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(img_input, (str, Path)):
            img = Image.open(img_input).convert("RGB")
            return np.asarray(img, dtype=np.float32) / 255.0
        elif isinstance(img_input, Image.Image):
            return np.asarray(img_input.convert("RGB"), dtype=np.float32) / 255.0
        elif isinstance(img_input, np.ndarray):
            arr = img_input.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[-1] > 3:
                arr = arr[..., :3]
            return arr
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

    @torch.inference_mode()
    def _predict_tile(self, tile1: np.ndarray, tile2: np.ndarray) -> np.ndarray:
        """Runs inference on a single 256x256 tile and returns 2D probability [0, 1]."""
        t1 = torch.from_numpy(tile1).permute(2, 0, 1)
        t2 = torch.from_numpy(tile2).permute(2, 0, 1)
        t1 = self.normalise(t1).unsqueeze(0).to(self.device)
        t2 = self.normalise(t2).unsqueeze(0).to(self.device)

        logits = self.model(t1, t2)[-1]
        probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
        return probs

    def predict(
        self,
        image_t1: Union[str, Path, Image.Image, np.ndarray],
        image_t2: Union[str, Path, Image.Image, np.ndarray],
        threshold: float = None,
    ) -> Dict[str, Any]:
        """
        Executes change detection on a pair of co-registered images.

        Returns a dictionary containing:
            - binary_mask: uint8 array (0 or 255) of shape (H, W)
            - probability_map: float32 array in [0, 1] of shape (H, W)
            - t1_rgb: uint8 array of shape (H, W, 3) for visualization
            - t2_rgb: uint8 array of shape (H, W, 3) for visualization
            - changed_pixels: int
            - total_pixels: int
            - change_percentage: float (0.0 to 100.0)
            - threshold: float used
        """
        thr = threshold if threshold is not None else self.threshold

        arr1 = self._load_image(image_t1)
        arr2 = self._load_image(image_t2)

        if arr1.shape[:2] != arr2.shape[:2]:
            raise ValueError(
                f"T1 and T2 must have identical dimensions, got {arr1.shape[:2]} vs {arr2.shape[:2]}"
            )

        H, W, _ = arr1.shape
        ts = self.tile_size
        stride = self.tile_stride

        # Case 1: Exact tile size (256x256)
        if H == ts and W == ts:
            prob_map = self._predict_tile(arr1, arr2)

        # Case 2: Undersized scene (< 256 in either dimension)
        elif H < ts or W < ts:
            pad_h = max(0, ts - H)
            pad_w = max(0, ts - W)
            padded1 = np.pad(arr1, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            padded2 = np.pad(arr2, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            full_prob = self._predict_tile(padded1[:ts, :ts], padded2[:ts, :ts])
            prob_map = full_prob[:H, :W]

        # Case 3: Larger scene (Sliding-window with smooth Hann blending)
        else:
            prob_accum = np.zeros((H, W), dtype=np.float32)
            weight_accum = np.zeros((H, W), dtype=np.float32)

            y_offsets = list(range(0, H - ts + 1, stride))
            if y_offsets[-1] != H - ts:
                y_offsets.append(H - ts)

            x_offsets = list(range(0, W - ts + 1, stride))
            if x_offsets[-1] != W - ts:
                x_offsets.append(W - ts)

            for y in y_offsets:
                for x in x_offsets:
                    t1_crop = arr1[y : y + ts, x : x + ts]
                    t2_crop = arr2[y : y + ts, x : x + ts]
                    tile_prob = self._predict_tile(t1_crop, t2_crop)

                    prob_accum[y : y + ts, x : x + ts] += tile_prob * self.window
                    weight_accum[y : y + ts, x : x + ts] += self.window

            prob_map = prob_accum / np.maximum(weight_accum, 1e-8)

        binary_mask = (prob_map >= thr).astype(np.uint8) * 255
        changed_pixels = int((binary_mask > 0).sum())
        total_pixels = int(H * W)
        change_pct = (changed_pixels / total_pixels) * 100.0

        return {
            "binary_mask": binary_mask,
            "probability_map": prob_map,
            "t1_rgb": (arr1 * 255).astype(np.uint8),
            "t2_rgb": (arr2 * 255).astype(np.uint8),
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(change_pct, 3),
            "threshold": thr,
        }

    def save_visualization(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        dpi: int = 150,
    ) -> Path:
        """
        Renders and saves a 4-panel visual verification sheet:
            [ T1 Before | T2 After | Change Probability Heatmap | T2 with Change Overlay ]
        """
        import matplotlib.pyplot as plt

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t1 = results["t1_rgb"]
        t2 = results["t2_rgb"]
        prob = results["probability_map"]
        mask = results["binary_mask"]
        pct = results["change_percentage"]
        thr = results["threshold"]

        # Create overlay on T2: Highlight changed areas in bright magenta/red
        overlay = t2.copy().astype(np.float32)
        change_indices = mask > 0
        overlay[change_indices, 0] = overlay[change_indices, 0] * 0.3 + 255 * 0.7  # Red
        overlay[change_indices, 1] = overlay[change_indices, 1] * 0.3              # Green
        overlay[change_indices, 2] = overlay[change_indices, 2] * 0.3 + 180 * 0.7  # Blue
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(t1)
        axes[0].set_title("T1 (Before)", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(t2)
        axes[1].set_title("T2 (After)", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        im = axes[2].imshow(prob, cmap="turbo", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"Confidence Heatmap (τ={thr:.2f})", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(overlay)
        axes[3].set_title(f"Change Overlay ({pct:.2f}% changed)", fontsize=12, fontweight="bold")
        axes[3].axis("off")

        fig.suptitle(
            f"ChangeFormer V3.1 Production Inference | {results['total_pixels']:,} px | {results['changed_pixels']:,} changed px ({pct:.2f}%)",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

__all__ = ["ChangeDetector", "DEFAULT_CHECKPOINT", "FROZEN_THRESHOLD"]
