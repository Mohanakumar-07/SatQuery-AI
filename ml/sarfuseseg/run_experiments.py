"""Run all six SAR-FuseSeg experiments end to end and write a combined report.

Usage (after ml/manifests/bigearthnet_subset_manifest.json exists):
    python -m ml.sarfuseseg.run_experiments
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.sarfuseseg.class_mapping import load_class_mapping
from ml.sarfuseseg.config import CHECKPOINT_DIR, MANIFEST_DIR
from ml.sarfuseseg.manifest import read_manifest
from ml.sarfuseseg.train import TrainConfig, train_one_experiment

EXPERIMENTS = ["optical_only", "sar_only", "fusion", "blank_optical", "blank_sar", "mismatched"]


def load_clc_id_to_name(path: Path) -> dict[int, str]:
    """CLC numeric-code -> label-name table (BigEarthNet ships this alongside the
    reference maps as CLC2018 code lists; see docs/ml/sarfuseseg_dataset.md)."""
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def main() -> None:
    manifest_path = MANIFEST_DIR / "bigearthnet_subset_manifest.json"
    entries = read_manifest(manifest_path)
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]

    mapping = load_class_mapping()
    clc_id_to_name = load_clc_id_to_name(MANIFEST_DIR / "clc_code_to_name.json")

    results = {}
    for experiment in EXPERIMENTS:
        config = TrainConfig(experiment_name=experiment)
        result = train_one_experiment(
            experiment,
            train_entries,
            val_entries,
            mapping,
            clc_id_to_name,
            config,
            output_dir=CHECKPOINT_DIR / experiment,
        )
        results[experiment] = result
        print(f"{experiment}: macro_f1={result['final_metrics']['macro_f1']:.4f} "
              f"mean_iou={result['final_metrics']['mean_iou']:.4f} "
              f"runtime={result['runtime_seconds']:.1f}s peak_gpu_mb={result['peak_gpu_mem_mb']}")

    (CHECKPOINT_DIR / "all_experiments_summary.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
