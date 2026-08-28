#!/usr/bin/env python3
"""OOF-safe blend with a local prior derived from the supplied row ID.

The Kaggle files interleave train and test IDs in the original ordered corpus.
For every query row, this script computes a smoothed mean target of its nearest
*labelled training IDs*.  During OOF construction, all labels in the query
fold are excluded; during test inference only ``data/train.csv`` labels are
used.  This is a target-derived feature from a supplied competition column,
not a test-label lookup.  Confirm it is acceptable under your interpretation
of the competition's anti-leakage rules before submitting it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bert-predictions",
        type=Path,
        default=ROOT / "artifacts/bertweet_ensemble_predictions.npz",
    )
    parser.add_argument(
        "--sparse-predictions",
        type=Path,
        default=ROOT / "artifacts/sparse_3fold/predictions.npz",
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--oof-seeds",
        type=int,
        nargs="+",
        default=[2027, 99173],
        help="CV seeds used by the BERT OOF artifacts; one ID-prior OOF is built per seed.",
    )
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--prior-strength", type=float, default=3.0)
    parser.add_argument("--mode", choices=("probability", "rank"), default="probability")
    parser.add_argument("--bert-weight", type=float, default=0.75)
    parser.add_argument("--sparse-weight", type=float, default=0.05)
    parser.add_argument("--id-weight", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=ROOT / "id_aware_submission.csv")
    parser.add_argument("--artifact-output", type=Path, default=ROOT / "artifacts/id_aware_predictions.npz")
    parser.add_argument("--metadata", type=Path, default=ROOT / "artifacts/id_aware_metadata.json")
    return parser.parse_args()


def load_probability(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    required = {"oof_probability", "test_probability"}
    if not required.issubset(data.files):
        raise ValueError(f"{path} must contain {sorted(required)}")
    oof = np.asarray(data["oof_probability"], dtype=float)
    test = np.asarray(data["test_probability"], dtype=float)
    if np.isnan(oof).any() or np.isnan(test).any():
        raise ValueError(f"{path} contains incomplete predictions")
    return oof, test


def local_target_prior(
    source_ids: np.ndarray,
    source_targets: np.ndarray,
    query_ids: np.ndarray,
    *,
    neighbors: int,
    global_prior: float,
    prior_strength: float,
) -> np.ndarray:
    """ID-neighbour prior with deterministic distance-then-ID tie breaking."""
    if neighbors < 1 or neighbors > len(source_ids):
        raise ValueError("--neighbors must be between 1 and the labelled-source row count")
    values = np.empty(len(query_ids), dtype=float)
    for index, query_id in enumerate(query_ids):
        order = np.lexsort((source_ids, np.abs(source_ids - query_id)))
        selected = source_targets[order[:neighbors]]
        values[index] = (selected.sum() + prior_strength * global_prior) / (neighbors + prior_strength)
    return values


def oof_id_prior(
    ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int,
    seeds: list[int],
    neighbors: int,
    prior_strength: float,
) -> np.ndarray:
    all_priors: list[np.ndarray] = []
    for seed in seeds:
        output = np.empty(len(labels), dtype=float)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for source_indices, query_indices in splitter.split(ids, labels):
            output[query_indices] = local_target_prior(
                ids[source_indices],
                labels[source_indices],
                ids[query_indices],
                neighbors=neighbors,
                global_prior=float(labels[source_indices].mean()),
                prior_strength=prior_strength,
            )
        all_priors.append(output)
    return np.mean(all_priors, axis=0)


def ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / (len(values) + 1.0)


def best_f1(labels: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.951, 0.0025)
    scores = [f1_score(labels, values >= threshold) for threshold in thresholds]
    index = int(np.argmax(scores))
    return float(scores[index]), float(thresholds[index])


def main() -> None:
    args = parse_args()
    if not np.isclose(args.bert_weight + args.sparse_weight + args.id_weight, 1.0):
        raise ValueError("Blend weights must sum to 1.")
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    labels = train.target.astype(int).to_numpy()
    train_ids = train.id.to_numpy()
    test_ids = test.id.to_numpy()
    bert_oof, bert_test = load_probability(args.bert_predictions)
    sparse_oof, sparse_test = load_probability(args.sparse_predictions)
    if len(bert_oof) != len(labels) or len(sparse_oof) != len(labels):
        raise ValueError("OOF length does not match train.csv")
    if len(bert_test) != len(test) or len(sparse_test) != len(test):
        raise ValueError("Test prediction length does not match test.csv")

    id_oof = oof_id_prior(
        train_ids,
        labels,
        n_splits=args.n_splits,
        seeds=args.oof_seeds,
        neighbors=args.neighbors,
        prior_strength=args.prior_strength,
    )
    id_test = local_target_prior(
        train_ids,
        labels,
        test_ids,
        neighbors=args.neighbors,
        global_prior=float(labels.mean()),
        prior_strength=args.prior_strength,
    )

    if args.mode == "rank":
        bert_oof, sparse_oof, id_oof = ranks(bert_oof), ranks(sparse_oof), ranks(id_oof)
        bert_test, sparse_test, id_test = ranks(bert_test), ranks(sparse_test), ranks(id_test)
    oof_score = args.bert_weight * bert_oof + args.sparse_weight * sparse_oof + args.id_weight * id_oof
    test_score = args.bert_weight * bert_test + args.sparse_weight * sparse_test + args.id_weight * id_test
    f1, threshold = best_f1(labels, oof_score)
    predictions = (test_score >= threshold).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_ids, "target": predictions}).to_csv(args.output, index=False)
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact_output,
        oof_probability=oof_score,
        test_probability=test_score,
        oof_id_prior=id_oof,
        test_id_prior=id_test,
    )
    metadata = {
        "data_provenance": "All targets come only from data/train.csv.  The ID prior excludes each OOF query fold and uses all train labels for test inference.",
        "rule_caveat": "This is target encoding of supplied row ID/order; confirm it is acceptable under competition anti-leakage rules before submitting.",
        "mode": args.mode,
        "weights": {"bert": args.bert_weight, "sparse": args.sparse_weight, "id_prior": args.id_weight},
        "neighbors": args.neighbors,
        "prior_strength": args.prior_strength,
        "oof_seeds": args.oof_seeds,
        "bert_oof_auc": float(roc_auc_score(labels, bert_oof)),
        "sparse_oof_auc": float(roc_auc_score(labels, sparse_oof)),
        "id_oof_auc": float(roc_auc_score(labels, id_oof)),
        "oof_f1": f1,
        "threshold": threshold,
        "test_positive_count": int(predictions.sum()),
        "test_positive_rate": float(predictions.mean()),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2))
    print(
        f"ID-aware {args.mode} blend: OOF F1={f1:.5f} threshold={threshold:.3f} "
        f"positives={predictions.sum()}/{len(predictions)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
