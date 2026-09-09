"""SatVLM vision-language adapter.

Handles scene description and visual question answering for optical and SAR imagery.
Implements the SpecialistAdapter contract (plan sections 4.4, 10.1, 11.0).

Hard limits (plan section 10.1):
    * never return coordinates or measured area — those come from masks
    * free-text confidence is evidence coverage + claim validation, not a percentage
"""

from __future__ import annotations

from app.models.base import AdapterProbe, AdapterRequest, AdapterResponse, BaseSpecialistAdapter


class SatVLMAdapter(BaseSpecialistAdapter):
    internal_name = "SatVLM"
    model_name = "Qwen2.5-VL-7B-Instruct"
    version = "satvlm-baseline-v1"
    preprocessing_version = "satvlm-preprocess-v1"
    requires_gpu = False
    checkpoint_hint = "satvlm"

    def available(self, *, capabilities: dict[str, bool] | None = None) -> AdapterProbe:
        return AdapterProbe(
            available=True,
            status="available",
            code="ADAPTER_READY",
            reason="Vision-language adapter is operational.",
        )

    def load(self, *, device: str | None = None) -> None:
        """No local checkpoint to load — inference is handled in the pipeline runner."""

    def infer(self, request: AdapterRequest) -> AdapterResponse:
        """Inference is performed by the pipeline runner, not directly through this adapter."""
        raise NotImplementedError(
            "SatVLMAdapter.infer() is not called directly. "
            "Inference runs through app.pipeline.runner.execute()."
        )

