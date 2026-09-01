"""Phase-1 acceptance script: run ChangeNet end to end on a real T1/T2 example.

Two runs, both on the same real LEVIR-CD sample pixels shipped with the vendored
ChangeFormer repo (docs/ml/changeformer_licence.md):

  A. As-is PNG pair (no georeferencing) -> pixel area / percent / relative location.
  B. The same pixels wrapped in a SYNTHETIC GeoTIFF (arbitrary but internally
     consistent UTM CRS + 0.5 m/px transform, chosen only to exercise the
     georeferenced code path — LEVIR-CD ships ~0.5 m/px imagery but not per-tile
     real-world coordinates, so this is explicitly a pipeline-validation fixture,
     not a claim about where this scene really is) -> polygons + area in m^2.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ml.changenet.adapter import ChangeNetAdapter
from ml.changenet.config import CHECKPOINT_DIR, LEVIR_CHECKPOINT_NAME, VENDOR_DIR

SAMPLE_DIR = VENDOR_DIR / "samples_LEVIR"
OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "reports" / "changenet_demo"

#: Synthetic fixture only (see module docstring) — arbitrary UTM zone 33N origin.
SYNTHETIC_CRS_EPSG = 32633
SYNTHETIC_ORIGIN = (500000.0, 4649776.0)  # documented arbitrary anchor, not a real scene location
SYNTHETIC_GSD_M = 0.5  # matches LEVIR-CD's published ~0.5 m/px resolution


def find_checkpoint() -> str:
    ckpt = CHECKPOINT_DIR / "best_ckpt.pt"
    if ckpt.exists():
        return str(ckpt)
    zip_path = CHECKPOINT_DIR / "CD_ChangeFormerV6_LEVIR.zip"
    with zipfile.ZipFile(zip_path) as zf:
        member = next(m for m in zf.namelist() if m.endswith(".pt"))
        zf.extract(member, CHECKPOINT_DIR)
        extracted = CHECKPOINT_DIR / member
        extracted.rename(ckpt)
    return str(ckpt)


def make_synthetic_geotiff(png_path: Path, out_path: Path) -> None:
    from PIL import Image

    arr = np.array(Image.open(png_path).convert("RGB"))
    h, w = arr.shape[:2]
    transform = from_origin(SYNTHETIC_ORIGIN[0], SYNTHETIC_ORIGIN[1], SYNTHETIC_GSD_M, SYNTHETIC_GSD_M)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=3, dtype=arr.dtype,
        crs=f"EPSG:{SYNTHETIC_CRS_EPSG}", transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(arr[:, :, band], band + 1)


def run_pair(adapter: ChangeNetAdapter, t1_path: str, t2_path: str, out_subdir: str) -> dict:
    adapter.validate(t1_path, t2_path)
    adapter.prepare()
    adapter.run()
    adapter.validate_output()
    result = adapter.to_result(t1_path, t2_path)
    paths = adapter.save_artifacts(str(OUT_DIR / out_subdir), result)
    return {"result": result.to_dict(), "artifact_paths": paths}


def main() -> None:
    checkpoint_path = find_checkpoint()
    adapter = ChangeNetAdapter(checkpoint_path, checkpoint_variant="LEVIR-CD")

    t1_png = SAMPLE_DIR / "A" / "test_7_0256_0512.png"
    t2_png = SAMPLE_DIR / "B" / "test_7_0256_0512.png"

    print("=== Path A: PNG pair, no georeferencing ===")
    out_a = run_pair(adapter, str(t1_png), str(t2_png), "path_a_png")
    print(json.dumps(out_a["result"], indent=2, default=str))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1_tif = OUT_DIR / "t1_synthetic.tif"
    t2_tif = OUT_DIR / "t2_synthetic.tif"
    make_synthetic_geotiff(t1_png, t1_tif)
    make_synthetic_geotiff(t2_png, t2_tif)

    print("\n=== Path B: synthetic-georeferenced GeoTIFF pair ===")
    out_b = run_pair(adapter, str(t1_tif), str(t2_tif), "path_b_geotiff")
    print(json.dumps(out_b["result"], indent=2, default=str))


if __name__ == "__main__":
    main()
