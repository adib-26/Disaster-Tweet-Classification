# Disaster Tweets Notebook Guide

This directory contains the primary modeling workflow for the Kaggle Natural Language Processing with Disaster Tweets competition. The notebooks are ordered as a modeling progression: baseline, sparse model, transformer model, probability ensemble, final conservative blend, and an experimental ID-aware blend.

Each notebook now includes professional visual diagnostics where they are most useful. These diagnostics cover dataset quality, engineered feature behavior, OOF probability separation, threshold search, component agreement, and blend selection.

## Recommended Run Order

1. `02_sparse_tfidf_ensemble.ipynb`
2. `03_bertweet_finetuning.ipynb`
3. `04_probability_ensemble.ipynb` if multiple BERTweet seed artifacts are available
4. `05_bertweet_sparse_blend.ipynb`

`01_original_stacked_baseline.ipynb` is retained as a classical reference baseline. `06_id_aware_blend_experiment.ipynb` is a separate research experiment and should be submitted only after confirming the competition rules allow the row-ID prior.

## Notebook Details

| Notebook | Role | Key outputs | Notes |
| --- | --- | --- | --- |
| `01_original_stacked_baseline.ipynb` | Classical NLP baseline with engineered features and a stacking or voting ensemble | `championship_submission.csv` | Useful for comparison and feature intuition |
| `02_sparse_tfidf_ensemble.ipynb` | Fast CPU-friendly sparse model using word and character TF-IDF | `sparse_submission.csv`, `artifacts/sparse*/predictions.npz` | Supplies sparse probabilities for blending |
| `03_bertweet_finetuning.ipynb` | Fine-tunes `vinai/bertweet-base` with stratified folds | `bertweet_submission.csv`, `artifacts/bertweet*/predictions.npz` | GPU recommended |
| `04_probability_ensemble.ipynb` | Averages compatible BERTweet probability artifacts | `bertweet_two_seed_submission.csv`, `artifacts/bertweet_ensemble_predictions.npz` | Improves prediction stability |
| `05_bertweet_sparse_blend.ipynb` | Selects BERTweet/sparse blend weight and threshold from OOF F1 | `best_submission.csv`, `artifacts/best_metadata.json` | Final conservative candidate workflow |
| `06_id_aware_blend_experiment.ipynb` | Adds an OOF-safe local row-ID prior | `id_aware_submission.csv`, `artifacts/id_aware_predictions.npz` | Experimental only |

## Visualization Coverage

| Notebook | Visual diagnostics |
| --- | --- |
| `01_original_stacked_baseline.ipynb` | Class balance, metadata coverage, tweet length distribution, highest-rate keywords |
| `02_sparse_tfidf_ensemble.ipynb` | Engineered numeric feature gaps, class-specific tweet length, sparse OOF threshold curve |
| `03_bertweet_finetuning.ipynb` | Dataset diagnostics, BERTweet OOF separation, threshold curve, fold stability |
| `04_probability_ensemble.ipynb` | Component probability distributions, OOF correlation heatmap, ensemble threshold curve |
| `05_bertweet_sparse_blend.ipynb` | Weight search curve, BERTweet/sparse agreement, final blend score separation |
| `06_id_aware_blend_experiment.ipynb` | ID prior behavior, component AUC comparison, ID-aware blend separation |

## Environment

Install dependencies from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The visualization cells use `matplotlib` and `seaborn`, which are declared in `requirements.txt`.

## Artifact Flow

```text
02_sparse_tfidf_ensemble.ipynb
  -> artifacts/sparse_3fold/predictions.npz

03_bertweet_finetuning.ipynb
  -> artifacts/bertweet_full/predictions.npz
  -> artifacts/bertweet_seed99173/predictions.npz, when running an additional seed

04_probability_ensemble.ipynb
  -> artifacts/bertweet_ensemble_predictions.npz

05_bertweet_sparse_blend.ipynb
  -> best_submission.csv
  -> artifacts/best_metadata.json

06_id_aware_blend_experiment.ipynb
  -> id_aware_submission.csv
  -> artifacts/id_aware_predictions.npz
```

Generated artifacts and submissions are ignored by Git and should be regenerated locally when needed.
