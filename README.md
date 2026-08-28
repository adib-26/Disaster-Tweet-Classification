# Kaggle Disaster Tweets

Rules-compliant modeling workspace for Kaggle's **Natural Language Processing with Disaster Tweets** competition.

The project contains classical sparse baselines, BERTweet fine-tuning, probability ensembling, and a final OOF-selected blend. The current strongest local candidate is generated as `best_submission.csv` from a two-seed BERTweet ensemble blended with a sparse TF-IDF model.

## Repository Layout

```text
.
|-- data/                         # Kaggle train.csv and test.csv
|-- notebooks/                    # Polished notebook versions for GitHub review
|-- bertweet_solution.py          # BERTweet cross-validation and inference pipeline
|-- sparse_solution.py            # Sparse TF-IDF ensemble pipeline
|-- ensemble_probabilities.py     # Average compatible OOF/test probability artifacts
|-- blend_submission.py           # OOF-selected BERTweet + sparse blend
|-- id_aware_blend.py             # Experimental OOF-safe ID-prior blend
|-- championship_solution.py      # Original stacked baseline
`-- requirements.txt
```

Generated files stay local and are gitignored: `artifacts/`, submission CSVs, `.venv/`, and `.hf_cache/`.

## Notebook Order

The notebooks are organized as a readable modeling story:

1. `notebooks/01_original_stacked_baseline.ipynb`
2. `notebooks/02_sparse_tfidf_ensemble.ipynb`
3. `notebooks/03_bertweet_finetuning.ipynb`
4. `notebooks/04_probability_ensemble.ipynb`
5. `notebooks/05_bertweet_sparse_blend.ipynb`
6. `notebooks/06_id_aware_blend_experiment.ipynb`

Notebook outputs are cleared for clean diffs. The matching `.py` scripts remain the source of truth for repeatable runs.

## Reproduce

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the sparse baseline:

```bash
.venv/bin/python sparse_solution.py \
  --n-splits 3 \
  --seed 2027 \
  --artifact-dir artifacts/sparse_3fold \
  --output sparse_submission_3fold.csv
```

Run a BERTweet fold or full run:

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

Blend the final probability artifacts:

```bash
.venv/bin/python blend_submission.py \
  --bert-predictions artifacts/bertweet_ensemble_predictions.npz \
  --sparse-predictions artifacts/sparse_3fold/predictions.npz \
  --score-mode probability \
  --output best_submission.csv \
  --metadata artifacts/best_metadata.json
```

## Validation Notes

- Thresholds are selected from out-of-fold predictions using F1.
- The pipelines do not read, infer, or hard-code test labels.
- `data/train.csv` and `data/test.csv` are included so the notebooks and scripts can run after clone.
- Generated assets such as `.venv/`, `.hf_cache/`, `artifacts/`, and submission CSVs are gitignored.
