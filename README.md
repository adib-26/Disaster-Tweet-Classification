# Kaggle Disaster Tweets

Rules-compliant modeling workspace for Kaggle's Natural Language Processing with Disaster Tweets competition.

This repository contains a complete NLP workflow for classifying whether tweets describe real disasters. The primary solution lives in the notebooks, progressing from classical sparse baselines to BERTweet fine-tuning and OOF-selected probability blending. The current strongest local candidate is `best_submission.csv`, generated from a BERTweet probability ensemble blended with a sparse TF-IDF model.

This solution achieved a top 50 ranking in the Kaggle competition.

## Quick Start

Create an environment and install the project dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Launch Jupyter:

```bash
.venv/bin/jupyter notebook
```

Recommended notebook workflow:

1. Run `notebooks/02_sparse_tfidf_ensemble.ipynb`.
2. Run `notebooks/03_bertweet_finetuning.ipynb`.
3. Run `notebooks/04_probability_ensemble.ipynb` if you have multiple BERTweet seed artifacts.
4. Run `notebooks/05_bertweet_sparse_blend.ipynb` to create the final conservative blend.

The notebooks include visual diagnostics for data quality, feature behavior, OOF probability separation, threshold search, model agreement, and blend selection.

## Repository Layout

```text
.
|-- data/
|   |-- train.csv
|   `-- test.csv
|-- notebooks/
|   |-- README.md
|   |-- 01_original_stacked_baseline.ipynb
|   |-- 02_sparse_tfidf_ensemble.ipynb
|   |-- 03_bertweet_finetuning.ipynb
|   |-- 04_probability_ensemble.ipynb
|   |-- 05_bertweet_sparse_blend.ipynb
|   `-- 06_id_aware_blend_experiment.ipynb
|-- championship_solution.py
|-- sparse_solution.py
|-- bertweet_solution.py
|-- ensemble_probabilities.py
|-- blend_submission.py
|-- id_aware_blend.py
|-- CHAMPIONSHIP_README.md
|-- README.md
`-- requirements.txt
```

Generated files are intentionally local and gitignored, including `.venv/`, `.hf_cache/`, `artifacts/`, and submission CSVs.

## Notebook Guide

| Notebook | Purpose | Main output |
| --- | --- | --- |
| `01_original_stacked_baseline.ipynb` | Classical text-cleaning, engineered-feature, and stacked-model baseline | `championship_submission.csv` |
| `02_sparse_tfidf_ensemble.ipynb` | Fast sparse TF-IDF model with OOF threshold selection | `sparse_submission.csv`, `artifacts/sparse*/predictions.npz` |
| `03_bertweet_finetuning.ipynb` | BERTweet cross-validation fine-tuning | `bertweet_submission.csv`, `artifacts/bertweet*/predictions.npz` |
| `04_probability_ensemble.ipynb` | Multi-seed probability averaging | `bertweet_two_seed_submission.csv`, `artifacts/bertweet_ensemble_predictions.npz` |
| `05_bertweet_sparse_blend.ipynb` | OOF-selected BERTweet and sparse-model blend | `best_submission.csv`, `artifacts/best_metadata.json` |
| `06_id_aware_blend_experiment.ipynb` | Experimental row-ID prior blend for research only | `id_aware_submission.csv`, `artifacts/id_aware_predictions.npz` |

## Script Alternatives

The notebooks are the preferred review format, but equivalent scripts are available for repeatable local runs.

Sparse baseline:

```bash
.venv/bin/python sparse_solution.py \
  --n-splits 3 \
  --seed 2027 \
  --artifact-dir artifacts/sparse_3fold \
  --output sparse_submission_3fold.csv
```

BERTweet fine-tuning:

```bash
.venv/bin/python bertweet_solution.py \
  --n-splits 3 \
  --epochs 3 \
  --batch-size 16 \
  --max-length 64 \
  --seed 2027 \
  --artifact-dir artifacts/bertweet_full \
  --output bertweet_submission.csv
```

Final blend:

```bash
.venv/bin/python blend_submission.py \
  --bert-predictions artifacts/bertweet_ensemble_predictions.npz \
  --sparse-predictions artifacts/sparse_3fold/predictions.npz \
  --score-mode probability \
  --output best_submission.csv \
  --metadata artifacts/best_metadata.json
```

## Validation Principles

- Thresholds are selected from out-of-fold predictions using F1.
- Blending weights are selected from OOF validation, not leaderboard feedback.
- The pipelines do not read, infer, or hard-code test labels.
- The ID-aware notebook is kept separate as an experiment and should be submitted only after reviewing the competition rules.
