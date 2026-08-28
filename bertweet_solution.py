#!/usr/bin/env python3
"""Rules-compliant BERTweet pipeline for Kaggle Disaster Tweets.

This script fine-tunes the public MIT-licensed ``vinai/bertweet-base`` model
only on ``data/train.csv``.  It does not retrieve, inspect, or use test
targets.  The final decision threshold is selected from out-of-fold training
predictions because the competition metric is F1 rather than accuracy.

Typical first run (the model must already be cached, or pass --allow-download):

    .venv/bin/python bertweet_solution.py \
        --n-splits 3 --epochs 3 --batch-size 16 --max-length 64

For a quick smoke test of a single validation fold:

    .venv/bin/python bertweet_solution.py \
        --n-splits 3 --only-fold 0 --epochs 1 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "vinai/bertweet-base"


@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    val_f1: float
    val_auc: float
    val_threshold: float
    seconds: float


class TweetDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: np.ndarray | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    # Hugging Face's default model cache lives under ``$HF_HOME/hub``.  Keeping
    # that final ``hub`` component here makes the explicit cache_dir compatible
    # with a cache populated via ``HF_HOME=<workspace>/.hf_cache``.
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".hf_cache" / "hub")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/bertweet")
    parser.add_argument("--output", type=Path, default=ROOT / "bertweet_submission.csv")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--only-fold", type=int, default=None, help="Run just this zero-indexed fold.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--freeze-bottom-layers", type=int, default=0)
    parser.add_argument(
        "--keyword-mode",
        choices=("none", "prefix", "pair"),
        default="none",
        help="How to expose the supplied keyword to BERTweet.  'none' is the validated default.",
    )
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads if the selected model is not cached.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for_run() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_license(model_name: str) -> str:
    if model_name.startswith("vinai/bertweet"):
        return "MIT (see Hugging Face model card)"
    if model_name.startswith("crisistransformers/CT-M3"):
        return "Apache-2.0 (see Hugging Face model card)"
    return "See the selected Hugging Face model card."


def tokenizer_for_model(args: argparse.Namespace):
    tokenizer_args = {
        "cache_dir": str(args.cache_dir),
        "local_files_only": not args.allow_download,
    }
    # BERTweet's author-provided tokenizer normalizes tweet-specific URL and
    # mention forms.  CT-M3 owns a distinct byte-level BPE tokenizer, so
    # passing BERTweet-only normalization options to it would be inappropriate.
    if args.model_name.startswith("vinai/bertweet"):
        tokenizer_args["normalization"] = True
    return AutoTokenizer.from_pretrained(args.model_name, **tokenizer_args)


def clean_optional(value: object) -> str:
    if pd.isna(value):
        return ""
    return unquote_plus(str(value)).strip()


def model_text(frame: pd.DataFrame, keyword_mode: str = "none") -> list[str]:
    """Keep the tweet intact and append the competition keyword as context.

    The supplied keyword is a feature available for both train and test.  We
    deliberately omit raw location: sparse-model OOF tests show it is noisy,
    whereas the keyword consistently adds signal.
    """
    tweets = frame["text"].fillna("").astype(str)
    if keyword_mode in {"none", "pair"}:
        return tweets.tolist()
    keywords = frame["keyword"].map(clean_optional)
    return [
        f"Keyword: {keyword}. Tweet: {tweet}" if keyword else tweet
        for tweet, keyword in zip(tweets, keywords, strict=True)
    ]


def keyword_pairs(frame: pd.DataFrame) -> list[str]:
    return [
        f"topic: {keyword}" if keyword else "topic: unknown"
        for keyword in frame["keyword"].map(clean_optional)
    ]


def encode_texts(
    tokenizer,
    texts: list[str],
    max_length: int,
    text_pairs: list[str] | None = None,
) -> dict[str, list[list[int]]]:
    return tokenizer(
        texts,
        text_pair=text_pairs,
        truncation="only_first" if text_pairs is not None else True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )


def make_loader(
    encodings: dict[str, list[list[int]]],
    labels: np.ndarray | None,
    collator: DataCollatorWithPadding,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TweetDataset(encodings, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=False,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


@torch.inference_mode()
def predict_probabilities(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        batch.pop("labels", None)
        logits = model(**batch).logits
        chunks.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(chunks)


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    # A fine, bounded grid is stable for this small competition and avoids
    # choosing a threshold from the test distribution.
    thresholds = np.arange(0.20, 0.801, 0.0025)
    scores = [f1_score(labels, probabilities >= threshold) for threshold in thresholds]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def freeze_lower_encoder_layers(model, count: int) -> None:
    if count <= 0:
        return
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.roberta.encoder.layer[:count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def train_fold(
    *,
    fold: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    labels: np.ndarray,
    train_encodings: dict[str, list[list[int]]],
    test_encodings: dict[str, list[list[int]]],
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, FoldResult]:
    set_seed(args.seed + fold)
    local_only = not args.allow_download
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        cache_dir=str(args.cache_dir),
        local_files_only=local_only,
    ).to(device)
    freeze_lower_encoder_layers(model, args.freeze_bottom_layers)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if device.type == "cuda" else None)
    train_loader = make_loader(
        {key: [value[i] for i in train_indices] for key, value in train_encodings.items()},
        labels[train_indices],
        collator,
        args.batch_size,
        True,
        args.num_workers,
    )
    validation_loader = make_loader(
        {key: [value[i] for i in validation_indices] for key, value in train_encodings.items()},
        labels[validation_indices],
        collator,
        args.batch_size,
        False,
        args.num_workers,
    )
    test_loader = make_loader(
        test_encodings,
        None,
        collator,
        args.batch_size,
        False,
        args.num_workers,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )

    best_validation_probability: np.ndarray | None = None
    best_test_probability: np.ndarray | None = None
    best_epoch = 0
    best_score = -np.inf
    best_threshold = 0.5
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch = move_batch(batch, device)
            result = model(**batch)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(result.loss.detach().cpu()))
            if step % 100 == 0 or step == len(train_loader):
                print(
                    f"fold={fold} epoch={epoch} batch={step}/{len(train_loader)} "
                    f"mean_loss={np.mean(losses):.4f}",
                    flush=True,
                )

        validation_probability = predict_probabilities(model, validation_loader, device)
        threshold, score = best_f1_threshold(labels[validation_indices], validation_probability)
        auc = roc_auc_score(labels[validation_indices], validation_probability)
        print(
            f"fold={fold} epoch={epoch} loss={np.mean(losses):.4f} "
            f"val_f1={score:.5f} val_auc={auc:.5f} threshold={threshold:.3f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_validation_probability = validation_probability
            best_test_probability = predict_probabilities(model, test_loader, device)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs > args.patience:
                print(f"fold={fold} early stop after epoch={epoch}", flush=True)
                break

    elapsed = time.perf_counter() - started
    assert best_validation_probability is not None and best_test_probability is not None
    result = FoldResult(
        fold=fold,
        best_epoch=best_epoch,
        val_f1=float(best_score),
        val_auc=float(roc_auc_score(labels[validation_indices], best_validation_probability)),
        val_threshold=float(best_threshold),
        seconds=float(elapsed),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_validation_probability, best_test_probability, result


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if args.only_fold is not None and not 0 <= args.only_fold < args.n_splits:
        raise ValueError("--only-fold must be between 0 and n_splits - 1")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    device = device_for_run()
    print(f"device={device.type} torch_threads={torch.get_num_threads()}", flush=True)

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    labels = train["target"].astype(int).to_numpy()
    print(f"train_rows={len(train)} test_rows={len(test)} positive_rate={labels.mean():.4f}", flush=True)

    # Passing local_files_only avoids accidental Hub calls once the approved
    # weights are cached.  --allow-download is an explicit opt-in for a fresh
    # setup.
    tokenizer = tokenizer_for_model(args)
    train_encodings = encode_texts(
        tokenizer,
        model_text(train, keyword_mode=args.keyword_mode),
        args.max_length,
        keyword_pairs(train) if args.keyword_mode == "pair" else None,
    )
    test_encodings = encode_texts(
        tokenizer,
        model_text(test, keyword_mode=args.keyword_mode),
        args.max_length,
        keyword_pairs(test) if args.keyword_mode == "pair" else None,
    )

    splitter = StratifiedKFold(args.n_splits, shuffle=True, random_state=args.seed)
    oof_probability = np.full(len(train), np.nan, dtype=np.float64)
    test_probability = np.zeros(len(test), dtype=np.float64)
    fold_results: list[FoldResult] = []
    completed_folds = 0

    for fold, (train_indices, validation_indices) in enumerate(splitter.split(train, labels)):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        print(
            f"\nstarting fold={fold} train={len(train_indices)} validation={len(validation_indices)}",
            flush=True,
        )
        validation_probability, fold_test_probability, result = train_fold(
            fold=fold,
            train_indices=train_indices,
            validation_indices=validation_indices,
            labels=labels,
            train_encodings=train_encodings,
            test_encodings=test_encodings,
            tokenizer=tokenizer,
            args=args,
            device=device,
        )
        oof_probability[validation_indices] = validation_probability
        test_probability += fold_test_probability
        fold_results.append(result)
        completed_folds += 1
        print(
            f"completed fold={fold} best_epoch={result.best_epoch} "
            f"val_f1={result.val_f1:.5f} elapsed={result.seconds / 60:.1f}m",
            flush=True,
        )

    if completed_folds == 0:
        raise RuntimeError("No folds were selected.")
    test_probability /= completed_folds

    metadata = {
        "model": args.model_name,
        "model_license": model_license(args.model_name),
        "data_provenance": "Only data/train.csv labels; no test targets or manual labels.",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "folds": [asdict(result) for result in fold_results],
    }

    # Single-fold mode is for smoke tests only: it deliberately avoids writing
    # a final submission whose threshold was selected from incomplete OOF data.
    if completed_folds != args.n_splits:
        (args.artifact_dir / "smoke_test_metadata.json").write_text(json.dumps(metadata, indent=2))
        np.savez_compressed(
            args.artifact_dir / "smoke_test_predictions.npz",
            oof_probability=oof_probability,
            test_probability=test_probability,
        )
        print("Single-fold smoke test complete; no submission written.", flush=True)
        return

    threshold, oof_f1 = best_f1_threshold(labels, oof_probability)
    oof_auc = roc_auc_score(labels, oof_probability)
    predictions = (test_probability >= threshold).astype(int)
    submission = pd.DataFrame({"id": test["id"], "target": predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    metadata.update(
        {
            "oof_f1": float(oof_f1),
            "oof_auc": float(oof_auc),
            "decision_threshold": float(threshold),
            "test_positive_count": int(predictions.sum()),
            "test_positive_rate": float(predictions.mean()),
            "output": str(args.output),
        }
    )
    (args.artifact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    np.savez_compressed(
        args.artifact_dir / "predictions.npz",
        oof_probability=oof_probability,
        test_probability=test_probability,
    )
    print(
        f"OOF F1={oof_f1:.5f} AUC={oof_auc:.5f} threshold={threshold:.3f} | "
        f"submission={args.output} positives={predictions.sum()}/{len(predictions)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
