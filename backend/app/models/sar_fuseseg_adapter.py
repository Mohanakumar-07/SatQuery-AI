"""SAR-FuseSeg adapter - CONTRACT ONLY, not implemented.

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
"""

from __future__ import annotations

from app.models.base import AdapterProbe, AdapterRequest, AdapterResponse, BaseSpecialistAdapter
from app.preprocessing.base import NotImplementedInContract


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

    def available(self, *, capabilities: dict[str, bool] | None = None) -> AdapterProbe:
        return AdapterProbe(
            available=False,
            status="not_implemented",
            code="ADAPTER_NOT_IMPLEMENTED",
            reason=(
                "SarFuseSegAdapter.infer() is a contract stub. Train and attach the "
                "dual-encoder model, then emit per-class masks with modality-specific "
                "preprocessing versions (plan section 10.3)."
            ),
        )

    def load(self, *, device: str | None = None) -> None:
        raise NotImplementedInContract("SarFuseSegAdapter.load() is not implemented.")

    def infer(self, request: AdapterRequest) -> AdapterResponse:
        raise NotImplementedInContract("SarFuseSegAdapter.infer() is not implemented.")
