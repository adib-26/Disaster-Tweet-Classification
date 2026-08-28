#!/usr/bin/env python3
"""Average compatible OOF/test probability artifacts and tune the F1 cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Completed *.npz artifacts with OOF and test scores.")
    parser.add_argument("--train-path", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "bertweet_ensemble_submission.csv")
    parser.add_argument(
        "--artifact-output", type=Path, default=ROOT / "artifacts/bertweet_ensemble_predictions.npz"
    )
    parser.add_argument(
        "--metadata", type=Path, default=ROOT / "artifacts/bertweet_ensemble_metadata.json"
    )
    return parser.parse_args()


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(path)
    if "oof_probability" not in values.files or "test_probability" not in values.files:
        raise ValueError(f"{path} is missing required probability arrays")
    oof = np.asarray(values["oof_probability"], dtype=float)
    test = np.asarray(values["test_probability"], dtype=float)
    if np.isnan(oof).any() or np.isnan(test).any():
        raise ValueError(f"{path} contains incomplete values")
    return oof, test


def best_f1(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.20, 0.801, 0.0025)
    scores = [f1_score(labels, probability >= threshold) for threshold in thresholds]
    index = int(np.argmax(scores))
    return float(scores[index]), float(thresholds[index])


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    labels = train.target.astype(int).to_numpy()
    arrays = [load(path) for path in args.inputs]
    if any(len(oof) != len(labels) or len(prediction) != len(test) for oof, prediction in arrays):
        raise ValueError("Artifact row counts do not match competition data.")
    oof_probability = np.mean([oof for oof, _ in arrays], axis=0)
    test_probability = np.mean([prediction for _, prediction in arrays], axis=0)
    score, threshold = best_f1(labels, oof_probability)
    predictions = (test_probability >= threshold).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test.id, "target": predictions}).to_csv(args.output, index=False)
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact_output, oof_probability=oof_probability, test_probability=test_probability)
    metadata = {
        "inputs": [str(path) for path in args.inputs],
        "data_provenance": "Only OOF predictions from models trained on data/train.csv labels.",
        "oof_f1": score,
        "oof_auc": float(roc_auc_score(labels, oof_probability)),
        "threshold": threshold,
        "test_positive_count": int(predictions.sum()),
        "test_positive_rate": float(predictions.mean()),
        "output": str(args.output),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2))
    print(
        f"Averaged {len(arrays)} artifacts: OOF F1={score:.5f} AUC={metadata['oof_auc']:.5f} "
        f"threshold={threshold:.3f}, positives={predictions.sum()}/{len(predictions)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
