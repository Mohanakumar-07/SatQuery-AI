"""ChangeNetAdapter — the ChangeFormer V6 specialist adapter (plan section 4.4).

ChangeFormer is ONLY a binary change detector in this MVP. This adapter may report
whether/where change occurred, how many regions, and changed area when georeferencing
exists. It must never claim a semantic land-cover transition (e.g. "vegetation became
built-up") — that is out of scope for ChangeNet and is left to SAR-FuseSeg / SatVLM in
later phases.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.changenet import postprocessing as post
from ml.changenet import preprocessing as pre
from ml.changenet.config import CHANGE_PROB_THRESHOLD, MIN_OVERLAP_PERCENT, TILE_SIZE
from ml.changenet.schemas import ModelInfo, SpecialistResult
from ml.changenet.vendor_bridge import (
    LoadedChangeFormer,
    load_changeformer_v6,
    predict_change_probability,
)

logger = logging.getLogger("changenet")


class ChangeNetAdapter:
    """validate() -> prepare() -> run() -> validate_output() -> to_result()."""

    def __init__(self, checkpoint_path: str, device: str | None = None, checkpoint_variant: str = "LEVIR-CD"):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_variant = checkpoint_variant
        self._loaded: LoadedChangeFormer | None = None
        self._device_arg = device
        self._warnings: list[str] = []
        self._log_lines: list[str] = []

    # ---- lifecycle -----------------------------------------------------------------
    def _ensure_loaded(self) -> LoadedChangeFormer:
        if self._loaded is None:
            self._log("loading ChangeFormerV6 checkpoint: %s" % self.checkpoint_path)
            self._loaded = load_changeformer_v6(self.checkpoint_path, device=self._device_arg)
            self._log("model loaded on device=%s" % self._loaded.device)
        return self._loaded

    def _log(self, msg: str) -> None:
        logger.info(msg)
        self._log_lines.append(msg)

    # ---- 1. validate -----------------------------------------------------------------
    def validate(self, t1_path: str, t2_path: str) -> dict[str, Any]:
        """Load both inputs and check they can legally be treated as a T1/T2 pair.

        Does not run the network. Raises pre.PreprocessingError on hard rejection
        (e.g. incompatible shapes with no georeferencing to reconcile them).
        """
        t1 = pre.load_source(t1_path)
        t2 = pre.load_source(t2_path)
        self._t1_source, self._t2_source = t1, t2
        return {
            "t1_georeferenced": t1.is_georeferenced,
            "t2_georeferenced": t2.is_georeferenced,
            "t1_shape": t1.array.shape[:2],
            "t2_shape": t2.array.shape[:2],
            "t1_crs_epsg": t1.crs_epsg,
            "t2_crs_epsg": t2.crs_epsg,
        }

    # ---- 2. prepare --------------------------------------------------------------
    def prepare(self) -> pre.CommonGridResult:
        """Common CRS/grid/resolution, residual alignment check, tiling."""
        grid = pre.prepare_common_grid(self._t1_source, self._t2_source)
        if not grid.alignment.passed:
            raise pre.PreprocessingError(
                f"residual alignment {grid.alignment.residual_offset_pixels}px exceeds "
                f"threshold {grid.alignment.threshold_pixels}px: {grid.alignment.reason}"
            )
        if grid.alignment.reason:
            self._warnings.append(f"alignment check: {grid.alignment.reason}")
        self._grid = grid
        self._tiles = pre.generate_tiles(grid, tile_size=TILE_SIZE)
        self._log(f"prepared {len(self._tiles)} tile(s) of {TILE_SIZE}x{TILE_SIZE}, georeferenced={grid.is_georeferenced}")
        return grid

    # ---- 3. run --------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        loaded = self._ensure_loaded()
        use_cuda = loaded.device.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(loaded.device)
        start = time.perf_counter()

        tile_probs: list[np.ndarray] = []
        for tile in self._tiles:
            t1_tensor = _to_tensor(tile.t1)
            t2_tensor = _to_tensor(tile.t2)
            prob = predict_change_probability(loaded, t1_tensor, t2_tensor)
            tile_probs.append(prob[0].cpu().numpy())

        elapsed_s = time.perf_counter() - start
        peak_mem_mb = float(torch.cuda.max_memory_allocated(loaded.device)) / (1024 * 1024) if use_cuda else None

        h, w = self._grid.t1.shape[:2]
        probability_map = post.assemble_probability_map(tile_probs, self._tiles, h, w)
        self._probability_map = probability_map
        self._runtime_seconds = elapsed_s
        self._peak_gpu_mem_mb = peak_mem_mb
        self._log(f"inference done in {elapsed_s:.3f}s, peak_gpu_mem_mb={peak_mem_mb}")
        return {"probability_map": probability_map, "runtime_seconds": elapsed_s, "peak_gpu_mem_mb": peak_mem_mb}

    # ---- 4. validate_output -----------------------------------------------------------
    def validate_output(self) -> dict[str, Any]:
        prob = self._probability_map
        checks = {
            "shape_matches_input": prob.shape == self._grid.t1.shape[:2],
            "values_in_unit_range": bool(np.all((prob >= 0.0) & (prob <= 1.0))),
            "not_nan": bool(not np.isnan(prob).any()),
        }
        if not all(checks.values()):
            raise ValueError(f"ChangeFormer output failed sanity checks: {checks}")

        clean_mask = post.clean_binary_mask(prob)
        labeled, regions = post.extract_regions(clean_mask)
        changed_fraction = float(clean_mask.mean())
        if changed_fraction > 0.9:
            self._warnings.append(
                f"{changed_fraction:.1%} of the scene flagged as changed; unusually high, "
                "check alignment/inputs before trusting this result"
            )
        self._clean_mask, self._labeled, self._regions = clean_mask, labeled, regions
        return {"changed_fraction": changed_fraction, "region_count": len(regions), "checks": checks}

    # ---- 5. to_result ------------------------------------------------------------------
    def to_result(self, t1_path: str, t2_path: str) -> SpecialistResult:
        grid = self._grid
        polygons = post.polygonize_mask(self._clean_mask, self._labeled, grid.transform, grid.crs_epsg)

        prediction: dict[str, Any] = {
            "change_detected": bool(self._clean_mask.any()),
            "region_count": len(self._regions),
            "image_size": {"height": grid.t1.shape[0], "width": grid.t1.shape[1]},
        }

        if grid.is_georeferenced:
            area = post.compute_area_m2(polygons)
            prediction["geographic_coordinates_available"] = True
            prediction["area_m2"] = area.total_area_m2
            prediction["measurement_crs_epsg"] = area.measurement_crs_epsg
            prediction["per_region_area_m2"] = area.per_region_area_m2
            if area.warning:
                self._warnings.append(area.warning)
            prediction["geojson"] = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"region_id": p.region_id, "pixel_area": p.pixel_area},
                        "geometry": p.geometry,
                    }
                    for p in polygons.polygons
                ],
            }
        else:
            pixel_area = post.compute_pixel_area(self._clean_mask, self._regions)
            prediction["geographic_coordinates_available"] = False
            prediction["pixel_area"] = pixel_area.total_pixels
            prediction["area_percent"] = pixel_area.total_pixels_percent
            prediction["relative_location"] = pixel_area.relative_location
            prediction["per_region_pixels"] = pixel_area.per_region_pixels
            self._warnings.append(
                "no georeferencing available on input(s): reporting pixel area/percentage/relative "
                "location only, per plan section 4.4 (no geographic coordinates or m^2 invented)"
            )

        evidence = {
            "t1_path": t1_path,
            "t2_path": t2_path,
            "alignment": asdict(grid.alignment),
            "cleanup_rule": post.CleanupRule().description,
            "tile_size": TILE_SIZE,
            "tile_count": len(self._tiles),
            "runtime_seconds": self._runtime_seconds,
            "peak_gpu_mem_mb": self._peak_gpu_mem_mb,
            "checkpoint_variant": self.checkpoint_variant,
        }

        confidence = {
            "changed_fraction": float(self._clean_mask.mean()),
            "mean_change_probability": float(self._probability_map.mean()),
            "alignment_confidence": grid.alignment.confidence,
        }

        result = SpecialistResult(
            task="change_detection",
            status="success",
            model=ModelInfo(name="ChangeNet", version=f"ChangeFormerV6-{self.checkpoint_variant}"),
            prediction=prediction,
            evidence=evidence,
            confidence=confidence,
            warnings=list(self._warnings),
        )
        return result

    # ---- artifact persistence (plan: save mask/polygons/stats/logs/runtime/gpu-mem) -------
    def save_artifacts(self, output_dir: str, result: SpecialistResult) -> dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}

        prob_path = out / "probability_mask.npy"
        np.save(prob_path, self._probability_map.astype(np.float32))
        paths["probability_mask"] = str(prob_path)

        from PIL import Image

        binary_png = out / "binary_mask.png"
        Image.fromarray((self._clean_mask * 255).astype(np.uint8)).save(binary_png)
        paths["binary_mask"] = str(binary_png)

        if "geojson" in result.prediction:
            geojson_path = out / "regions.geojson"
            geojson_path.write_text(json.dumps(result.prediction["geojson"], indent=2))
            paths["geojson"] = str(geojson_path)

        stats_path = out / "statistics.json"
        stats_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        paths["statistics"] = str(stats_path)

        log_path = out / "inference.log"
        log_path.write_text("\n".join(self._log_lines))
        paths["log"] = str(log_path)

        return paths


def _to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 3, H, W) float tensor, normalized like the upstream dataset."""
    from ml.changenet.config import NORM_MEAN, NORM_STD

    arr = rgb_uint8.astype(np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std = np.array(NORM_STD, dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor
