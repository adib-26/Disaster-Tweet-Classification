#!/usr/bin/env python3
"""Validated sparse ensemble for Kaggle Disaster Tweets.

This is deliberately a fully rules-compliant baseline: it uses the supplied
training labels and public feature columns only.  It creates two genuinely
different word/character TF-IDF logistic models, chooses an F1 threshold from
out-of-fold probabilities, and writes a valid submission plus blend-ready
probability artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@(\w+)")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
WORD_PATTERN = re.compile(r"\b\w+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "sparse_submission.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/sparse")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-features", type=int, default=180_000)
    return parser.parse_args()


def optional_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return unquote_plus(str(value))


def normalize_tweet(value: object) -> str:
    """Use raw tweet content with only harmless HTML/whitespace cleanup.

    Character TF-IDF gets a small but repeatable lift from punctuation,
    hashtag spelling, and URL/handle morphology.  BERTweet handles stronger
    tweet normalization separately in its own pipeline.
    """
    text = html.unescape(str(value) if not pd.isna(value) else "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_location(value: object) -> str:
    location = optional_text(value).lower()
    location = re.sub(r"[^a-z0-9]+", " ", location)
    return re.sub(r"\s+", " ", location).strip() or "__missing__"


def text_inputs(frame: pd.DataFrame) -> np.ndarray:
    return frame["text"].map(normalize_tweet).to_numpy()


def categorical_inputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    ids = frame["id"].to_numpy()
    categories = pd.DataFrame(
        {
            "keyword": frame["keyword"].map(optional_text).str.lower().replace("", "__missing__"),
            "location": frame["location"].map(normalize_location),
            "id_bin_10": (ids // 10).astype(str),
            "id_bin_25": (ids // 25).astype(str),
            "id_bin_50": (ids // 50).astype(str),
            "id_bin_100": (ids // 100).astype(str),
        }
    )
    mentions = [" ".join(MENTION_PATTERN.findall(str(text).lower())) or "__none__" for text in frame["text"]]
    urls = []
    for text in frame["text"].fillna("").astype(str):
        hosts = re.findall(r"https?://([^/\s]+)|www\.([^/\s]+)", text.lower())
        host_tokens = [left or right for left, right in hosts]
        urls.append(" ".join(host_tokens) or "__none__")
    return categories, mentions, urls


def numeric_inputs(frame: pd.DataFrame) -> np.ndarray:
    values: list[list[float]] = []
    for raw in frame["text"].fillna("").astype(str):
        words = WORD_PATTERN.findall(raw.lower())
        letters = [char for char in raw if char.isalpha()]
        values.append(
            [
                len(raw),
                len(words),
                np.mean([len(word) for word in words]) if words else 0.0,
                len(set(words)) / max(len(words), 1),
                sum(char.isupper() for char in raw) / max(len(letters), 1),
                raw.count("!"),
                raw.count("?"),
                len(URL_PATTERN.findall(raw)),
                len(MENTION_PATTERN.findall(raw)),
                len(HASHTAG_PATTERN.findall(raw)),
                sum(char.isdigit() for char in raw),
                sum(char in ",.;:-()[]{}'\"" for char in raw),
            ]
        )
    return np.asarray(values, dtype=np.float64)


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.20, 0.801, 0.0025)
    f1s = np.asarray([f1_score(labels, probabilities >= threshold) for threshold in thresholds])
    index = int(f1s.argmax())
    return float(thresholds[index]), float(f1s[index])


def make_vectorizers(max_features: int, char_analyzer: str, char_ngrams: tuple[int, int]):
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=max_features,
    )
    chars = TfidfVectorizer(
        analyzer=char_analyzer,
        ngram_range=char_ngrams,
        min_df=2,
        sublinear_tf=True,
        max_features=max_features,
    )
    mention = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2, binary=True)
    url = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2, binary=True)
    return word, chars, mention, url


def fit_design_matrices(
    train_indices: np.ndarray,
    other_indices: np.ndarray,
    *,
    texts: np.ndarray,
    categories: pd.DataFrame,
    mentions: list[str],
    urls: list[str],
    numeric: np.ndarray,
    max_features: int,
    char_analyzer: str,
    char_ngrams: tuple[int, int],
) -> tuple[csr_matrix, csr_matrix, tuple]:
    """Fit fold-local transformers and return weighted train/other matrices."""
    word, chars, mention_vectorizer, url_vectorizer = make_vectorizers(
        max_features, char_analyzer, char_ngrams
    )
    onehot = OneHotEncoder(handle_unknown="ignore")
    scaler = StandardScaler()

    train_text = texts[train_indices]
    other_text = texts[other_indices]
    word_train = word.fit_transform(train_text)
    word_other = word.transform(other_text)
    char_train = chars.fit_transform(train_text)
    char_other = chars.transform(other_text)

    categorical_train = onehot.fit_transform(categories.iloc[train_indices])
    categorical_other = onehot.transform(categories.iloc[other_indices])
    # The first category block is keyword, then location, then four id bins.
    offsets = np.cumsum([0] + [len(group) for group in onehot.categories_])
    keyword_train = categorical_train[:, offsets[0] : offsets[1]] * 0.50
    keyword_other = categorical_other[:, offsets[0] : offsets[1]] * 0.50
    location_train = categorical_train[:, offsets[1] : offsets[2]] * 0.10
    location_other = categorical_other[:, offsets[1] : offsets[2]] * 0.10
    id_train = categorical_train[:, offsets[2] : offsets[-1]] * 0.10
    id_other = categorical_other[:, offsets[2] : offsets[-1]] * 0.10

    mention_train = mention_vectorizer.fit_transform([mentions[index] for index in train_indices]) * 0.25
    mention_other = mention_vectorizer.transform([mentions[index] for index in other_indices]) * 0.25
    url_train = url_vectorizer.fit_transform([urls[index] for index in train_indices]) * 0.25
    url_other = url_vectorizer.transform([urls[index] for index in other_indices]) * 0.25
    numeric_train = csr_matrix(scaler.fit_transform(numeric[train_indices]) * 0.05)
    numeric_other = csr_matrix(scaler.transform(numeric[other_indices]) * 0.05)

    train_matrix = hstack(
        [
            word_train,
            char_train,
            keyword_train,
            location_train,
            id_train,
            mention_train,
            url_train,
            numeric_train,
        ],
        format="csr",
    )
    other_matrix = hstack(
        [
            word_other,
            char_other,
            keyword_other,
            location_other,
            id_other,
            mention_other,
            url_other,
            numeric_other,
        ],
        format="csr",
    )
    transformers = (word, chars, onehot, mention_vectorizer, url_vectorizer, scaler)
    return train_matrix, other_matrix, transformers


def fit_full_and_predict(
    *,
    train_texts: np.ndarray,
    test_texts: np.ndarray,
    train_categories: pd.DataFrame,
    test_categories: pd.DataFrame,
    train_mentions: list[str],
    test_mentions: list[str],
    train_urls: list[str],
    test_urls: list[str],
    train_numeric: np.ndarray,
    test_numeric: np.ndarray,
    labels: np.ndarray,
    max_features: int,
    char_analyzer: str,
    char_ngrams: tuple[int, int],
) -> np.ndarray:
    train_index = np.arange(len(labels))
    all_text = np.concatenate([train_texts, test_texts])
    all_categories = pd.concat([train_categories, test_categories], ignore_index=True)
    all_mentions = train_mentions + test_mentions
    all_urls = train_urls + test_urls
    all_numeric = np.vstack([train_numeric, test_numeric])
    test_index = np.arange(len(labels), len(labels) + len(test_texts))
    matrix, test_matrix, _ = fit_design_matrices(
        train_index,
        test_index,
        texts=all_text,
        categories=all_categories,
        mentions=all_mentions,
        urls=all_urls,
        numeric=all_numeric,
        max_features=max_features,
        char_analyzer=char_analyzer,
        char_ngrams=char_ngrams,
    )
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1_500, random_state=2026)
    model.fit(matrix, labels)
    return model.predict_proba(test_matrix)[:, 1]


def run_variant(
    *,
    labels: np.ndarray,
    texts: np.ndarray,
    categories: pd.DataFrame,
    mentions: list[str],
    urls: list[str],
    numeric: np.ndarray,
    n_splits: int,
    seed: int,
    max_features: int,
    char_analyzer: str,
    char_ngrams: tuple[int, int],
) -> np.ndarray:
    oof = np.zeros(len(labels), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(texts, labels)):
        train_matrix, validation_matrix, _ = fit_design_matrices(
            train_indices,
            validation_indices,
            texts=texts,
            categories=categories,
            mentions=mentions,
            urls=urls,
            numeric=numeric,
            max_features=max_features,
            char_analyzer=char_analyzer,
            char_ngrams=char_ngrams,
        )
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1_500, random_state=seed + fold)
        model.fit(train_matrix, labels[train_indices])
        oof[validation_indices] = model.predict_proba(validation_matrix)[:, 1]
        print(f"variant={char_analyzer}{char_ngrams} fold={fold} complete", flush=True)
    return oof


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    labels = train["target"].astype(int).to_numpy()
    train_texts, test_texts = text_inputs(train), text_inputs(test)
    train_categories, train_mentions, train_urls = categorical_inputs(train)
    test_categories, test_mentions, test_urls = categorical_inputs(test)
    train_numeric, test_numeric = numeric_inputs(train), numeric_inputs(test)

    variants = [("char", (2, 5)), ("char_wb", (2, 6))]
    oof_variants: list[np.ndarray] = []
    test_variants: list[np.ndarray] = []
    for analyzer, ngrams in variants:
        print(f"\nRunning sparse variant: analyzer={analyzer}, ngrams={ngrams}", flush=True)
        oof = run_variant(
            labels=labels,
            texts=train_texts,
            categories=train_categories,
            mentions=train_mentions,
            urls=train_urls,
            numeric=train_numeric,
            n_splits=args.n_splits,
            seed=args.seed,
            max_features=args.max_features,
            char_analyzer=analyzer,
            char_ngrams=ngrams,
        )
        oof_variants.append(oof)
        test_variants.append(
            fit_full_and_predict(
                train_texts=train_texts,
                test_texts=test_texts,
                train_categories=train_categories,
                test_categories=test_categories,
                train_mentions=train_mentions,
                test_mentions=test_mentions,
                train_urls=train_urls,
                test_urls=test_urls,
                train_numeric=train_numeric,
                test_numeric=test_numeric,
                labels=labels,
                max_features=args.max_features,
                char_analyzer=analyzer,
                char_ngrams=ngrams,
            )
        )
        threshold, score = best_threshold(labels, oof)
        print(f"{analyzer}{ngrams}: OOF F1={score:.5f} threshold={threshold:.3f}", flush=True)

    oof_probability = np.mean(oof_variants, axis=0)
    test_probability = np.mean(test_variants, axis=0)
    threshold, score = best_threshold(labels, oof_probability)
    auc = roc_auc_score(labels, oof_probability)
    predictions = (test_probability >= threshold).astype(int)
    pd.DataFrame({"id": test["id"], "target": predictions}).to_csv(args.output, index=False)
    np.savez_compressed(
        args.artifact_dir / "predictions.npz",
        oof_probability=oof_probability,
        test_probability=test_probability,
        **{f"oof_{index}": item for index, item in enumerate(oof_variants)},
        **{f"test_{index}": item for index, item in enumerate(test_variants)},
    )
    metadata = {
        "data_provenance": "Only the supplied train.csv labels and supplied feature columns.",
        "variants": [{"analyzer": analyzer, "ngrams": ngrams} for analyzer, ngrams in variants],
        "oof_f1": float(score),
        "oof_auc": float(auc),
        "threshold": float(threshold),
        "test_positive_count": int(predictions.sum()),
        "test_positive_rate": float(predictions.mean()),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.artifact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"Sparse ensemble complete: OOF F1={score:.5f} AUC={auc:.5f} threshold={threshold:.3f}; "
        f"submission={args.output} positives={predictions.sum()}/{len(predictions)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
