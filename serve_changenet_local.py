"""serve_changenet_local.py -- Local GPU inference server for ChangeFormer V6 change detection.

Exposes an HTTP interface on port 8002 so Render can dispatch change detection
queries over the Cloudflare Tunnel (changenet.ettalabs.in).

Usage:
  python serve_changenet_local.py [--port 8002]

Then in a terminal (combined with satvlm-tunnel config):
  cloudflared tunnel --config "C:/Users/kmoha/.cloudflared/satvlm-config.yml" run satvlm-tunnel
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
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
logger = logging.getLogger("serve_changenet_local")

app = FastAPI(title="SatQuery Local ChangeFormer Gateway", version="1.0.0")
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
    from app.models.changenet_adapter import ChangeNetAdapter
    adapter = ChangeNetAdapter()
    adapter.load()
    _ADAPTER = adapter
    logger.info("ChangeFormer adapter loaded")
    return _ADAPTER


class ImageB64(BaseModel):
    label: str
    base64: str
    filename: str = "image.tif"


class ChangeNetRequest(BaseModel):
    analysis_id: Optional[str] = None
    before: ImageB64
    after: ImageB64
    transform: Optional[Any] = None
    crs: Optional[str] = None


@app.get("/")
@app.get("/health")
def health():
    try:
        from app.models.changenet_adapter import ChangeNetAdapter
        probe = ChangeNetAdapter().available()
        return {
            "status": "healthy" if probe.available else "degraded",
            "service": "SatQuery Local ChangeFormer Gateway",
            "model": "ChangeFormer-V6",
            "adapter_available": probe.available,
            "adapter_reason": probe.reason,
        }
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


@app.post("/infer")
def infer(req: ChangeNetRequest):
    import tempfile, json
    try:
        adapter = get_adapter()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ChangeFormer not available: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        before_path = tmp_path / req.before.filename
        after_path = tmp_path / req.after.filename
        before_path.write_bytes(base64.b64decode(req.before.base64))
        after_path.write_bytes(base64.b64decode(req.after.base64))
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        # Build a minimal AdapterRequest-compatible dict for direct impl call
        bundle_dict = {
            "t1_image": str(before_path),
            "t2_image": str(after_path),
            "transform": req.transform,
            "crs": req.crs,
        }
        try:
            result = adapter._impl.execute(bundle_dict)
        except Exception as exc:
            logger.error("ChangeFormer inference error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "source": "ChangeNet",
        "version": "V3.1-Champion",
        "evidence": result.get("evidence", {}),
        "prediction": result.get("prediction", {}),
        "trace": ["changenet_local_gateway", "changeformer_v6_inferred"],
    }


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    try:
        get_adapter()
    except Exception as e:
        logger.warning("Could not pre-load ChangeFormer (will retry on first request): %s", e)

    logger.info("Starting ChangeFormer gateway on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
