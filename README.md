# Kaggle Disaster Tweets

A rules-compliant NLP pipeline for Kaggle's [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) competition, classifying whether a tweet describes a real disaster.

**Result:** top 50 on the leaderboard, from `best_submission.csv`, a BERTweet probability ensemble blended with a sparse TF-IDF model.

## Approach

The solution progresses through four stages, each with its own notebook and script:

1. **Sparse baseline** – TF-IDF features with out-of-fold threshold selection.
2. **BERTweet fine-tuning** – cross-validated transformer fine-tuning on the full training set.
3. **Multi-seed ensemble** – probability averaging across BERTweet seeds to reduce variance.
4. **Final blend** – an OOF-selected combination of the BERTweet ensemble and the sparse model, which produced the best local score.

A fifth, experimental notebook explores a row-ID-aware prior. It's kept separate from the main pipeline since it depends on how strictly you want to interpret the competition's rules on ID leakage, see [Validation Principles](#validation-principles).

`CHAMPIONSHIP_README.md` documents the earlier classical baseline (text cleaning, engineered features, stacked models) that seeded this approach before the BERTweet stages were added.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter notebook
```

Run the notebooks in order:

1. `notebooks/02_sparse_tfidf_ensemble.ipynb`
2. `notebooks/03_bertweet_finetuning.ipynb`
3. `notebooks/04_probability_ensemble.ipynb` (needs multiple BERTweet seed artifacts)
4. `notebooks/05_bertweet_sparse_blend.ipynb` for the final blend

Each notebook includes diagnostics: data quality, feature behavior, OOF probability separation, threshold search, model agreement, and blend selection.

## Repository Layout

```text
.
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── README.md
│   ├── 01_original_stacked_baseline.ipynb
│   ├── 02_sparse_tfidf_ensemble.ipynb
│   ├── 03_bertweet_finetuning.ipynb
│   ├── 04_probability_ensemble.ipynb
│   ├── 05_bertweet_sparse_blend.ipynb
│   └── 06_id_aware_blend_experiment.ipynb
├── README.md
└── requirements.txt
```

Generated files (`.venv/`, `.hf_cache/`, `artifacts/`, submission CSVs) are gitignored and rebuilt locally.

## Notebook Guide

| Notebook | Purpose | Main output |
| --- | --- | --- |
| `01_original_stacked_baseline.ipynb` | Classical text-cleaning, engineered-feature, and stacked-model baseline | `championship_submission.csv` |
| `02_sparse_tfidf_ensemble.ipynb` | Fast sparse TF-IDF model with OOF threshold selection | `sparse_submission.csv`, `artifacts/sparse*/predictions.npz` |
| `03_bertweet_finetuning.ipynb` | BERTweet cross-validation fine-tuning | `bertweet_submission.csv`, `artifacts/bertweet*/predictions.npz` |
| `04_probability_ensemble.ipynb` | Multi-seed probability averaging | `bertweet_two_seed_submission.csv`, `artifacts/bertweet_ensemble_predictions.npz` |
| `05_bertweet_sparse_blend.ipynb` | OOF-selected BERTweet + sparse blend | `best_submission.csv`, `artifacts/best_metadata.json` |
| `06_id_aware_blend_experiment.ipynb` | Experimental row-ID prior blend, research only | `id_aware_submission.csv`, `artifacts/id_aware_predictions.npz` |

## Script Alternatives

The notebooks are the preferred review format. Equivalent scripts are available for repeatable local runs.

**Sparse baseline**

```bash
.venv/bin/python sparse_solution.py \
  --n-splits 3 \
  --seed 2027 \
  --artifact-dir artifacts/sparse_3fold \
  --output sparse_submission_3fold.csv
```

**BERTweet fine-tuning**

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

**Final blend**

```bash
.venv/bin/python blend_submission.py \
  --bert-predictions artifacts/bertweet_ensemble_predictions.npz \
  --sparse-predictions artifacts/sparse_3fold/predictions.npz \
  --score-mode probability \
  --output best_submission.csv \
  --metadata artifacts/best_metadata.json
```
## Result Preview
<img width="928" height="295" alt="Screenshot 2026-08-28 at 4 06 53 PM" src="https://github.com/user-attachments/assets/8ed7365f-c0e5-4406-af2c-42c6f8010dc9" />


## Validation Principles

- Thresholds are selected from out-of-fold predictions using F1, never the leaderboard.
- Blend weights come from OOF validation, not leaderboard feedback.
- The pipelines never read, infer, or hard-code test labels.
- The ID-aware notebook (06) is kept separate as an experiment. Review the competition's rules on ID-based leakage before submitting anything derived from it.
