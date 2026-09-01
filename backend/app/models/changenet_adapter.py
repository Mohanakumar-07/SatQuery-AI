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

from app.models.base import AdapterProbe, AdapterRequest, AdapterResponse, BaseSpecialistAdapter
from app.preprocessing.base import NotImplementedInContract


class ChangeNetAdapter(BaseSpecialistAdapter):
    internal_name = "ChangeNet"
    model_name = "ChangeFormer-V6"
    version = "baseline-v1"
    preprocessing_version = "changenet-preprocess-v1"
    requires_gpu = True
    checkpoint_hint = "changenet"

    def available(self, *, capabilities: dict[str, bool] | None = None) -> AdapterProbe:
        return AdapterProbe(
            available=False,
            status="not_implemented",
            code="ADAPTER_NOT_IMPLEMENTED",
            reason=(
                "ChangeNetAdapter.infer() is a contract stub. Wire the pretrained "
                "ChangeFormer V6 checkpoint plus paired-tile preprocessing and the "
                "inverse transform back to scene coordinates (plan section 10.2)."
            ),
        )

    def load(self, *, device: str | None = None) -> None:
        raise NotImplementedInContract("ChangeNetAdapter.load() is not implemented.")

    def infer(self, request: AdapterRequest) -> AdapterResponse:
        raise NotImplementedInContract("ChangeNetAdapter.infer() is not implemented.")
