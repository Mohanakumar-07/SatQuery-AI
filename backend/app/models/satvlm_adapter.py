"""SatVLM adapter - CONTRACT ONLY, not implemented.

Owner: member 1 (SatVLM integration and QLoRA).

Contract to satisfy (plan sections 4.4 SatVLM adapter, 10.1, 11.0):
    * render optical inputs as RGB or a documented false-colour composite
    * render SAR inputs from calibrated VV/VH with a documented visualisation transform
    * tile large scenes and preserve the tile -> scene mapping
    * apply Qwen-specific resizing and a maximum visual token limit
    * record rendering recipe, bands, tile coordinates, processor version and
      maximum pixel configuration in every response

Hard limits:
    * never return coordinates or measured area - those come from masks (section 9)
    * free-text confidence is evidence coverage + claim validation, not a percentage
    * the MVP answer must come from the QLoRA-adapted checkpoint, not the generic one
"""

from __future__ import annotations

from app.models.base import AdapterProbe, AdapterRequest, AdapterResponse, BaseSpecialistAdapter
from app.preprocessing.base import NotImplementedInContract


class SatVLMAdapter(BaseSpecialistAdapter):
    internal_name = "SatVLM"
    model_name = "Qwen2.5-VL-7B-Instruct"
    version = "satvlm-baseline-v1"
    preprocessing_version = "satvlm-preprocess-v1"
    requires_gpu = True
    checkpoint_hint = "satvlm"

    def available(self, *, capabilities: dict[str, bool] | None = None) -> AdapterProbe:
        return AdapterProbe(
            available=False,
            status="not_implemented",
            code="ADAPTER_NOT_IMPLEMENTED",
            reason=(
                "SatVLMAdapter.infer() is a contract stub. Implement Qwen2.5-VL loading, "
                "the documented optical/SAR rendering recipe and tiling before the VQA "
                "workflow can run (plan section 10.1)."
            ),
        )

    def load(self, *, device: str | None = None) -> None:
        raise NotImplementedInContract("SatVLMAdapter.load() is not implemented.")

    def infer(self, request: AdapterRequest) -> AdapterResponse:
        raise NotImplementedInContract("SatVLMAdapter.infer() is not implemented.")
