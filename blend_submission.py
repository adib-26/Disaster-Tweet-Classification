#!/usr/bin/env python3
"""Blend OOF-validated BERTweet and sparse probability artifacts.

The script never reads target data for the test set.  It learns only a global
blend weight and F1 threshold from the two models' out-of-fold predictions,
then applies the selected rule to their averaged test probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bert-predictions",
        type=Path,
        default=ROOT / "artifacts/bertweet_full/predictions.npz",
    )
    parser.add_argument(
        "--sparse-predictions",
        type=Path,
        default=ROOT / "artifacts/sparse_3fold/predictions.npz",
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "blended_submission.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "artifacts/blend_metadata.json")
    parser.add_argument(
        "--score-mode",
        choices=("rank", "probability"),
        default="rank",
        help="Rank scores are safer when model probability calibration differs.",
    )
    return parser.parse_args()


def load_probability_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    required = {"oof_probability", "test_probability"}
    if not required.issubset(data.files):
        raise ValueError(f"{path} must contain {sorted(required)}; found {data.files}")
    oof = np.asarray(data["oof_probability"], dtype=float)
    test = np.asarray(data["test_probability"], dtype=float)
    if np.isnan(oof).any() or np.isnan(test).any():
        raise ValueError(f"{path} contains incomplete predictions; finish all folds before blending.")
    return oof, test


def to_ranks(values: np.ndarray) -> np.ndarray:
    # Mid-ranks preserve ties and map the smallest/largest value into (0, 1).
    return rankdata(values, method="average") / (len(values) + 1.0)


def best_f1(labels: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.951, 0.0025)
    f1s = np.asarray([f1_score(labels, values >= threshold) for threshold in thresholds])
    index = int(f1s.argmax())
    return float(f1s[index]), float(thresholds[index])


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    labels = train["target"].astype(int).to_numpy()
    bert_oof, bert_test = load_probability_pair(args.bert_predictions)
    sparse_oof, sparse_test = load_probability_pair(args.sparse_predictions)
    if not (len(labels) == len(bert_oof) == len(sparse_oof)):
        raise ValueError("OOF prediction lengths must match data/train.csv.")
    if not (len(test) == len(bert_test) == len(sparse_test)):
        raise ValueError("Test prediction lengths must match data/test.csv.")

    if args.score_mode == "rank":
        bert_oof_score, sparse_oof_score = to_ranks(bert_oof), to_ranks(sparse_oof)
        bert_test_score, sparse_test_score = to_ranks(bert_test), to_ranks(sparse_test)
    else:
        bert_oof_score, sparse_oof_score = bert_oof, sparse_oof
        bert_test_score, sparse_test_score = bert_test, sparse_test

    candidates: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    for bert_weight in np.arange(0.50, 1.001, 0.025):
        oof_score = bert_weight * bert_oof_score + (1.0 - bert_weight) * sparse_oof_score
        f1, threshold = best_f1(labels, oof_score)
        candidate = {"bert_weight": float(bert_weight), "f1": f1, "threshold": threshold}
        candidates.append(candidate)
        if best is None or candidate["f1"] > best["f1"]:
            best = candidate
    assert best is not None

    final_score = best["bert_weight"] * bert_test_score + (1.0 - best["bert_weight"]) * sparse_test_score
    predictions = (final_score >= best["threshold"]).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"], "target": predictions}).to_csv(args.output, index=False)

    metadata = {
        "data_provenance": "OOF blend selection used only data/train.csv labels; no test targets or manual labels.",
        "score_mode": args.score_mode,
        "bert_oof_auc": float(roc_auc_score(labels, bert_oof)),
        "sparse_oof_auc": float(roc_auc_score(labels, sparse_oof)),
        "selected": best,
        "candidates": candidates,
        "test_positive_count": int(predictions.sum()),
        "test_positive_rate": float(predictions.mean()),
        "output": str(args.output),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2))
    print(
        f"Blend complete: OOF F1={best['f1']:.5f} threshold={best['threshold']:.3f} "
        f"BERT weight={best['bert_weight']:.3f} positives={predictions.sum()}/{len(predictions)} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
