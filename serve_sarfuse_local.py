"""serve_sarfuse_local.py -- Local GPU inference server for SAR-FuseSeg V3.

Exposes an HTTP interface on port 8003 so Render can dispatch SAR fusion
queries over the Cloudflare Tunnel (sarfuse.ettalabs.in).

Usage:
  python serve_sarfuse_local.py [--port 8003]
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("serve_sarfuse_local")

app = FastAPI(title="SatQuery Local SAR-FuseSeg Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ADAPTER = None


def get_adapter():
    global _ADAPTER
    if _ADAPTER is not None:
        return _ADAPTER
    from app.models.sar_fuseseg_adapter import SarFuseSegAdapter
    adapter = SarFuseSegAdapter()
    adapter.load()
    _ADAPTER = adapter
    logger.info("SAR-FuseSeg adapter loaded")
    return _ADAPTER


class ImageB64(BaseModel):
    label: str
    base64: str
    filename: str = "image.tif"


class SARFuseRequest(BaseModel):
    analysis_id: Optional[str] = None
    optical: ImageB64
    sar: ImageB64
    transform: Optional[Any] = None
    crs: Optional[str] = None


@app.get("/")
@app.get("/health")
def health():
    try:
        from app.models.sar_fuseseg_adapter import SarFuseSegAdapter
        probe = SarFuseSegAdapter().available()
        return {
            "status": "healthy" if probe.available else "degraded",
            "service": "SatQuery Local SAR-FuseSeg Gateway",
            "model": "SAR-FuseSeg-V3",
            "adapter_available": probe.available,
            "adapter_reason": probe.reason,
        }
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


@app.post("/infer")
def infer(req: SARFuseRequest):
    try:
        adapter = get_adapter()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"SAR-FuseSeg not available: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        optical_path = tmp_path / req.optical.filename
        sar_path = tmp_path / req.sar.filename
        optical_path.write_bytes(base64.b64decode(req.optical.base64))
        sar_path.write_bytes(base64.b64decode(req.sar.base64))
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        import rasterio
        import numpy as np
        with rasterio.open(optical_path) as ds_opt:
            optical_arr = ds_opt.read()
            transform = ds_opt.transform
            crs = ds_opt.crs
        with rasterio.open(sar_path) as ds_sar:
            sar_arr = ds_sar.read()

        bundle_dict = {
            "optical": optical_arr,
            "sar": sar_arr,
            "transform": transform,
            "crs": crs,
        }
        try:
            facts = adapter._impl.infer(bundle_dict)
        except Exception as exc:
            logger.error("SAR-FuseSeg inference error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "source": "SAR-FuseSeg",
        "version": "V3",
        "geojson": getattr(facts, "geojson", None),
        "facts": facts.to_dict() if hasattr(facts, "to_dict") else {},
        "trace": ["sarfuse_local_gateway", "sar_fuseseg_v3_inferred"],
    }


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    try:
        get_adapter()
    except Exception as e:
        logger.warning("Could not pre-load SAR-FuseSeg (will retry on first request): %s", e)

    logger.info("Starting SAR-FuseSeg gateway on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
