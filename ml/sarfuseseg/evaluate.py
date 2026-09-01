"""Segmentation metrics: Macro-F1, mean IoU, per-class IoU, per-class recall.

All metrics are computed from a confusion matrix that excludes ``ignore_index``
pixels entirely (never counted as any class, never counted as background).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.sarfuseseg.config import CLASS_NAMES, IGNORE_INDEX, N_CLASSES


@dataclass
class ConfusionMatrix:
    n_classes: int = N_CLASSES
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64))

    def update(self, pred: np.ndarray, target: np.ndarray, ignore_index: int = IGNORE_INDEX) -> None:
        valid = target != ignore_index
        pred, target = pred[valid], target[valid]
        idx = target.astype(np.int64) * self.n_classes + pred.astype(np.int64)
        counts = np.bincount(idx, minlength=self.n_classes ** 2)
        self.matrix += counts.reshape(self.n_classes, self.n_classes)


@dataclass(frozen=True)
class Metrics:
    macro_f1: float
    mean_iou: float
    per_class_iou: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_precision: dict[str, float]
    confusion_matrix: list[list[int]]


def compute_metrics(cm: ConfusionMatrix, class_names: list[str] = CLASS_NAMES) -> Metrics:
    m = cm.matrix.astype(np.float64)
    tp = np.diag(m)
    fp = m.sum(axis=0) - tp
    fn = m.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn), np.nan)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)

    return Metrics(
        macro_f1=float(np.nanmean(f1)),
        mean_iou=float(np.nanmean(iou)),
        per_class_iou={name: (float(v) if not np.isnan(v) else None) for name, v in zip(class_names, iou)},
        per_class_recall={name: (float(v) if not np.isnan(v) else None) for name, v in zip(class_names, recall)},
        per_class_precision={name: (float(v) if not np.isnan(v) else None) for name, v in zip(class_names, precision)},
        confusion_matrix=cm.matrix.tolist(),
    )
