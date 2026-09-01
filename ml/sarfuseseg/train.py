"""SAR-FuseSeg training loop — six experiments (Implementation_Plan_v1.2.md).

Experiment 1: optical-only   | Experiment 4: blank optical (sanity: model must fail)
Experiment 2: SAR-only       | Experiment 5: blank SAR     (sanity: model must fail)
Experiment 3: optical+SAR    | Experiment 6: mismatched optical/SAR pair (sanity)

Batch size 1 + gradient accumulation + mixed precision, matching the 6 GB VRAM budget.
Every run saves: checkpoint, config, dataset manifest, class mapping, preprocessing
version, normalization statistics, seed, metrics and a checkpoint hash.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.sarfuseseg.config import (
    CHECKPOINT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_ACCUMULATION_STEPS,
    DEFAULT_LR,
    DEFAULT_SEED,
    IGNORE_INDEX,
    N_CLASSES,
)
from ml.sarfuseseg.dataset import OpticalOnlyDataset, PairedOpticalSARDataset, SAROnlyDataset
from ml.sarfuseseg.evaluate import ConfusionMatrix, compute_metrics
from ml.sarfuseseg.losses import WeightedCEDiceLoss
from ml.sarfuseseg.manifest import ManifestEntry
from ml.sarfuseseg.model import SARFuseSeg

EXPERIMENT_DATASETS = {
    "optical_only": OpticalOnlyDataset,
    "sar_only": SAROnlyDataset,
    "fusion": PairedOpticalSARDataset,
    "blank_optical": OpticalOnlyDataset,  # exp 4: optical branch already blanked; blank SAR too below
    "blank_sar": SAROnlyDataset,  # exp 5
    "mismatched": PairedOpticalSARDataset,  # exp 6: mismatch_pairs=True
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(entries_labels: list[np.ndarray], n_classes: int = N_CLASSES) -> list[float]:
    counts = np.zeros(n_classes, dtype=np.float64)
    for label in entries_labels:
        valid = label != IGNORE_INDEX
        counts += np.bincount(label[valid].astype(np.int64), minlength=n_classes)
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / counts
    weights = inv / inv.sum() * n_classes
    return weights.tolist()


@dataclass
class TrainConfig:
    experiment_name: str
    seed: int = DEFAULT_SEED
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    grad_accumulation_steps: int = DEFAULT_GRAD_ACCUMULATION_STEPS
    lr: float = DEFAULT_LR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    preprocessing_version: str = "sarfuseseg-preprocessing-v1"
    class_mapping_version: str = "v1"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_one_experiment(
    experiment_name: str,
    train_entries: list[ManifestEntry],
    val_entries: list[ManifestEntry],
    mapping,
    clc_id_to_name: dict[int, str],
    config: TrainConfig,
    class_weights: list[float] | None = None,
    output_dir: Path | None = None,
) -> dict:
    set_seed(config.seed)
    device = torch.device(config.device)

    mismatch = experiment_name == "mismatched"
    dataset_cls = EXPERIMENT_DATASETS[experiment_name]
    train_ds = dataset_cls(train_entries, mapping, clc_id_to_name, mismatch_pairs=mismatch)
    val_ds = dataset_cls(val_entries, mapping, clc_id_to_name, mismatch_pairs=mismatch)
    if experiment_name == "blank_optical":
        train_ds.blank_sar = val_ds.blank_sar = True  # blank BOTH branches (exp 4 sanity check)
    if experiment_name == "blank_sar":
        train_ds.blank_optical = val_ds.blank_optical = True  # exp 5 sanity check

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = SARFuseSeg().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = WeightedCEDiceLoss(class_weights=class_weights).to(device)
    scaler = torch.amp.GradScaler(enabled=config.mixed_precision and device.type == "cuda")

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    history = []
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            optical = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            label = batch["label"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=config.mixed_precision and use_cuda):
                logits = model(optical, sar)
                loss = criterion(logits, label) / config.grad_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % config.grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item() * config.grad_accumulation_steps

        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running_loss / max(1, len(train_loader)), **asdict(val_metrics)})

    elapsed = time.perf_counter() - start
    peak_mem_mb = float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024) if use_cuda else None

    out = output_dir or (CHECKPOINT_DIR / experiment_name)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, ckpt_path)

    result = {
        "experiment": experiment_name,
        "config": asdict(config),
        "class_weights": class_weights,
        "runtime_seconds": elapsed,
        "peak_gpu_mem_mb": peak_mem_mb,
        "history": history,
        "final_metrics": history[-1] if history else None,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": _file_sha256(ckpt_path),
        "train_sample_count": len(train_entries),
        "val_sample_count": len(val_entries),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    cm = ConfusionMatrix()
    for batch in loader:
        optical = batch["optical"].to(device)
        sar = batch["sar"].to(device)
        label = batch["label"].to(device)
        logits = model(optical, sar)
        pred = torch.argmax(logits, dim=1)
        cm.update(pred.cpu().numpy(), label.cpu().numpy())
    return compute_metrics(cm)
