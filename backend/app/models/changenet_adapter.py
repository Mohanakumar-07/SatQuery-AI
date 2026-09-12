"""ChangeNet (ChangeFormer V6) adapter - CONTRACT ONLY, not implemented.

Owner: member 3 (ChangeNet and SAR-FuseSeg).

Contract to satisfy (plan sections 4.4 ChangeNet adapter, 10.2):
    * reproject T1/T2 onto the bundle's common CRS, grid, resolution and extent
    * validate residual alignment and REJECT the pair beyond the validated tolerance
    * generate identical spatial crops for both dates, then fixed-size paired tiles
    * apply the normalisation the selected ChangeFormer checkpoint expects
    * restore the mask to original coordinates with the recorded inverse transform
    * emit binary change facts only

Hard limits:
    * binary change only - never assert that a region became built-up, water or
      vegetation (sections 10.2 and 22)
    * noise removal only through a documented, validated post-processing policy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from app.models.base import AdapterProbe, AdapterRequest, AdapterResponse, BaseSpecialistAdapter
from app.preprocessing.base import NotImplementedInContract

# Dynamic search for repo root to import changeformer and evidence_engine
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


class ChangeNetAdapter(BaseSpecialistAdapter):
    internal_name = "ChangeNet"
    model_name = "ChangeFormer-V6"
    version = "V3.1-Champion"
    preprocessing_version = "changenet-preprocess-v1"
    requires_gpu = False
    checkpoint_hint = "changenet"

    def __init__(self, *, settings: Any | None = None, params: dict[str, Any] | None = None) -> None:
        super().__init__(settings=settings, params=params)
        self._impl = None

    def _resolve_checkpoint(self) -> Path | None:
        candidates = [
            _REPO_ROOT / "artifacts" / "checkpoints" / "changeformer" / "changeformer_v3_1_best.pt",
            _REPO_ROOT / "artifacts" / "checkpoints" / "changeformer" / "changeformer_levir_best.pt",
            _REPO_ROOT / "checkpoints" / "changeformer_v3_1_best.pt",
            _REPO_ROOT / "checkpoints" / "changeformer_levir" / "best_ckpt.pt",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def available(self, *, capabilities: dict[str, bool] | None = None) -> AdapterProbe:
        try:
            import torch  # noqa: F401
            import einops  # noqa: F401
        except ImportError as exc:
            return AdapterProbe(
                available=False,
                status="unavailable",
                code="DEPENDENCY_MISSING",
                reason=f"ChangeFormer ML dependency missing ({exc.name}). Runs in GPU worker environment.",
            )
        ckpt = self._resolve_checkpoint()
        if ckpt is None:
            return AdapterProbe(
                available=False,
                status="unavailable",
                code="CHECKPOINT_NOT_FOUND",
                reason="ChangeFormer V3.1 checkpoint not found under artifacts/checkpoints/changeformer/.",
            )
        return AdapterProbe(
            available=True,
            status="available",
            code="ADAPTER_READY",
            reason=f"ChangeFormer V3.1 champion available at {ckpt.name}.",
            loaded=self._impl is not None,
        )

    def load(self, *, device: str | None = None) -> None:
        if self._impl is not None:
            return
        ckpt = self._resolve_checkpoint()
        if ckpt is None:
            raise FileNotFoundError("ChangeFormer V3.1 checkpoint not found.")
        try:
            from changeformer.adapter import ChangeNetAdapter as ImplAdapter
            self._impl = ImplAdapter(
                checkpoint_path=ckpt,
                device=device or "cpu",
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load ChangeFormer specialist: {exc}") from exc

    def infer(self, request: AdapterRequest) -> AdapterResponse:
        if self._impl is None:
            self.load()

        before_src = request.bundle.source_for("before")
        after_src = request.bundle.source_for("after")
        if not before_src or not after_src:
            raise ValueError("SceneBundle must have 'before' and 'after' sources for ChangeNet.")

        bundle_dict = {
            "t1_image": str(before_src.path),
            "t2_image": str(after_src.path),
            "transform": before_src.transform,
            "crs": before_src.crs,
        }

        result = self._impl.execute(bundle_dict)
        request.work_dir.mkdir(parents=True, exist_ok=True)

        geojson_path = request.work_dir / "change_features.geojson"
        geojson_data = result.get("evidence", {}).get("geojson")
        if geojson_data:
            geojson_path.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")

        return AdapterResponse(
            source=self.internal_name,
            version=self.version,
            mask_paths=[],
            geojson_paths=[geojson_path] if geojson_data else [],
            facts=result.get("evidence", {}),
            raw_score=result.get("prediction", {}).get("change_percentage"),
            score_kind="change_percentage",
            answer_status="detected" if result.get("prediction", {}).get("changed_pixels", 0) > 0 else "no_change",
            trace=["changeformer_v6_inferred", "evidence_engine_extracted"],
        )
