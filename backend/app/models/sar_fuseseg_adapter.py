"""SAR-FuseSeg adapter - CONTRACT ONLY, not implemented.
"""
SAR-FuseSeg V3 backend specialist adapter.

Owner: member 3 (ChangeNet and SAR-FuseSeg).

Frozen MVP baseline the adapter must match (plan section 10.3):
    * optical input: RGB or the explicitly selected Sentinel-2 bands of the dataset
    * SAR input: VV and VH, calibrated and log/decibel scaled as selected
    * encoders: ResNet-18 (optical) + ResNet-18 (SAR), multi-scale concatenation fusion
    * decoder: U-Net style; classes built_up, water, vegetation, other + one ignore index
    * loss: weighted cross-entropy + Dice; class weights from the frozen training split
    * tile size 256x256 until the baseline is reproducible

Hard limits:
    * separate normalisation statistics per modality, and a valid-data mask for
      nodata/border/invalid pixels
    * report optical-only, SAR-only and fused results side by side - never averaged
Encapsulates SAR-FuseSeg V3 (ResNet-18 dual-encoder + multi-scale fusion + U-Net decoder)
with strict input validation, frozen 120x120 direct / versioned tiled inference, and Evidence Engine grounding.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from app.models.base import AdapterProbe, AdapterRequest, AdapterResponse, BaseSpecialistAdapter
from app.preprocessing.base import NotImplementedInContract

# Dynamic search for repo root to import sar_fuse_seg and evidence_engine
_curr = Path(__file__).resolve()
_REPO_ROOT = None
for _p in _curr.parents:
    if (_p / "artifacts").is_dir() or (_p / ".git").is_dir():
        _REPO_ROOT = _p
        break
if _REPO_ROOT is None:
    _REPO_ROOT = _curr.parents[3]

for _sub in ["src", "ml"]:
    _sp = str(_REPO_ROOT / _sub)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)


class SarFuseSegAdapter(BaseSpecialistAdapter):
    internal_name = "SAR-FuseSeg"
    model_name = "resnet18-dual-encoder-unet"
    version = "fuseseg-v0"
    preprocessing_version = "sar-fuseseg-preprocess-v1"
    requires_gpu = True
    checkpoint_hint = "sar_fuseseg"
    #: Fixed by the plan; changing it is an experiment, not an MVP default.
    classes: tuple[str, ...] = ("built_up", "water", "vegetation", "other")
    tile_size = 256
    model_name = "SAR-FuseSeg-V3"
    version = "V3"
    preprocessing_version = "sar_fuse_v3_preprocess"
    requires_gpu = False
    checkpoint_hint = "sar_fuse_v3"
    classes: tuple[str, ...] = ("built-up", "water", "vegetation")

    def __init__(self, *, settings: Any | None = None, params: dict[str, Any] | None = None) -> None:
        super().__init__(settings=settings, params=params)
        self._impl = None

    def _resolve_checkpoint(self) -> Path | None:
        candidates = [
            _REPO_ROOT / "models" / "checkpoints" / "sar_fuse_v3" / "best.pt",
            _REPO_ROOT / "sar-fuse-V3" / "best.pt",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def available(self, *, capabilities: dict[str, bool] | None = None) -> AdapterProbe:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError as exc:
            return AdapterProbe(
                available=False,
                status="unavailable",
                code="DEPENDENCY_MISSING",
                reason=f"PyTorch / TorchVision dependency missing ({exc.name}).",
            )
        ckpt = self._resolve_checkpoint()
        if ckpt is None:
            return AdapterProbe(
                available=False,
                status="unavailable",
                code="CHECKPOINT_NOT_FOUND",
                reason="SAR-FuseSeg V3 checkpoint not found under models/checkpoints/sar_fuse_v3/.",
            )
        return AdapterProbe(
            available=False,
            status="not_implemented",
            code="ADAPTER_NOT_IMPLEMENTED",
            reason=(
                "SarFuseSegAdapter.infer() is a contract stub. Train and attach the "
                "dual-encoder model, then emit per-class masks with modality-specific "
                "preprocessing versions (plan section 10.3)."
            ),
            available=True,
            status="available",
            code="ADAPTER_READY",
            reason=f"SAR-FuseSeg V3 champion available at {ckpt.name}.",
            loaded=self._impl is not None,
        )

    def load(self, *, device: str | None = None) -> None:
        raise NotImplementedInContract("SarFuseSegAdapter.load() is not implemented.")
        if self._impl is not None:
            return
        ckpt = self._resolve_checkpoint()
        if ckpt is None:
            raise FileNotFoundError("SAR-FuseSeg V3 checkpoint not found.")
        try:
            from sar_fuse_seg.adapter import SARFuseSegAdapter as ImplAdapter
            self._impl = ImplAdapter(
                checkpoint_path=ckpt,
                device=device or "cpu",
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load SAR-FuseSeg specialist: {exc}") from exc

    def _remote_url(self) -> str | None:
        url = os.environ.get("SATQUERY_SARFUSE_URL", "").strip().rstrip("/")
        return url if url else None

    def infer(self, request: AdapterRequest) -> AdapterResponse:
        import base64, httpx

        remote = self._remote_url()
        opt_src = request.bundle.source_for("optical") or request.bundle.source_for("primary")
        sar_src = request.bundle.source_for("sar") or request.bundle.source_for("secondary")
        if not opt_src or not sar_src:
            raise ValueError("SceneBundle must contain both 'optical' and 'sar' sources for SAR-FuseSeg V3.")

        if remote:
            logger.info("Delegating SAR-FuseSeg inference to remote gateway: %s", remote)
            optical_b64 = base64.b64encode(Path(opt_src.path).read_bytes()).decode()
            sar_b64 = base64.b64encode(Path(sar_src.path).read_bytes()).decode()
            payload = {
                "analysis_id": getattr(request, "analysis_id", None),
                "optical": {"label": "optical", "base64": optical_b64, "filename": Path(opt_src.path).name},
                "sar": {"label": "sar", "base64": sar_b64, "filename": Path(sar_src.path).name},
                "crs": str(opt_src.crs) if opt_src.crs else None,
            }
            try:
                resp = httpx.post(f"{remote}/infer", json=payload, timeout=300)
                resp.raise_for_status()
                result = resp.json()
            except Exception as exc:
                raise RuntimeError(f"Remote SAR-FuseSeg gateway error: {exc}") from exc

            geojson_data = result.get("geojson")
            facts_dict = result.get("facts", {})
            trace = result.get("trace", ["sar_fuseseg_v3_inferred"])
        else:
            # Local in-process (stub — model must be implemented first)
            raise NotImplementedInContract("SarFuseSegAdapter.infer() local path not implemented. Set SATQUERY_SARFUSE_URL to use the tunnel gateway.")

        request.work_dir.mkdir(parents=True, exist_ok=True)
        geojson_path = request.work_dir / "landcover_features.geojson"
        if geojson_data:
            geojson_path.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")

        return AdapterResponse(
            source=self.internal_name,
            version=self.version,
            mask_paths=[],
            geojson_paths=[geojson_path] if geojson_data else [],
            facts=facts_dict,
            raw_score=None,
            score_kind="class_distribution",
            answer_status="success",
            trace=trace,
        )

