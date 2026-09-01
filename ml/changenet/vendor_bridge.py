"""Bridge to the vendored ChangeFormer network (docs/ml/changeformer_licence.md).

Loads ``ChangeFormerV6`` directly from ``ml/vendor/ChangeFormer`` and applies a
downloaded checkpoint. We build the network ourselves instead of using the upstream
``models.networks.define_G`` / ``CDEvaluator`` plumbing so we can feed our own
preprocessed tensors and get a soft probability map, not just the upstream demo's
argmax*255 visualisation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

from ml.changenet.config import EMBED_DIM, INPUT_CHANNELS, N_CLASSES, VENDOR_DIR


def _ensure_vendor_on_path() -> None:
    vendor_str = str(VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


@dataclass
class LoadedChangeFormer:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: str
    best_val_acc: float | None
    best_epoch_id: int | None


def load_changeformer_v6(checkpoint_path: str, device: str | None = None) -> LoadedChangeFormer:
    """Build ``ChangeFormerV6`` and load a ``main_cd.py``-style training checkpoint.

    The checkpoint dict is expected to contain ``model_G_state_dict`` (and, for the
    official releases, ``best_val_acc`` / ``best_epoch_id``), matching
    ``models/basic_model.py::CDEvaluator.load_checkpoint`` in the vendored repo.
    """
    _ensure_vendor_on_path()
    from models.ChangeFormer import ChangeFormerV6  # type: ignore  # vendored module

    resolved_device = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    net = ChangeFormerV6(
        input_nc=INPUT_CHANNELS,
        output_nc=N_CLASSES,
        decoder_softmax=False,
        embed_dim=EMBED_DIM,
    )

    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    state_dict = checkpoint["model_G_state_dict"] if "model_G_state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.to(resolved_device)
    net.eval()

    return LoadedChangeFormer(
        model=net,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        best_val_acc=checkpoint.get("best_val_acc") if isinstance(checkpoint, dict) else None,
        best_epoch_id=checkpoint.get("best_epoch_id") if isinstance(checkpoint, dict) else None,
    )


@torch.no_grad()
def predict_change_probability(loaded: LoadedChangeFormer, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Run the network and return the *change-class* probability map.

    ``t1``/``t2``: ``[B, 3, H, W]`` tensors, already normalized (config.NORM_MEAN/STD).
    Returns: ``[B, H, W]`` float tensor in ``[0, 1]``.
    """
    t1 = t1.to(loaded.device)
    t2 = t2.to(loaded.device)
    outputs = loaded.model(t1, t2)
    logits = outputs[-1]  # final, full-resolution scale (models/ChangeFormer.py)
    probs = torch.softmax(logits, dim=1)
    return probs[:, 1, :, :]
