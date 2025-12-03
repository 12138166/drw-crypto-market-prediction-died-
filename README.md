# DRW - Crypto Market Prediction -died

This repository contains an end-to-end solution for the Kaggle competition  
**“DRW - Crypto Market Prediction”**, focusing on building a robust tree-based ensemble
for crypto market return prediction.

> Note: The competition data is not included in this repository.  
> You need to download it from Kaggle and update the paths in `config.py` if necessary.

---

## Project Overview

The main goals of this project are:

- To build a **fully reproducible pipeline** for the DRW crypto market prediction task.
- To explore how different **training-window lengths** (full history vs. recent data)
  and **time-decay sample weights** affect predictive performance.
- To combine multiple **gradient boosting models** (XGBoost and LightGBM) and
  multiple **data slices** via ensembling, evaluated by **Pearson correlation** between
  predictions and the target.

The code is written to be reusable for other financial time-series prediction tasks.

---

## Methodology

### Features and Target

The model uses a subset of the provided numeric features (e.g. `X863`, `X856`, ...),
plus order book / volume variables such as `bid_qty`, `ask_qty`, `buy_qty`,
`sell_qty`, and `volume`.  
The target variable is the column `label` in the training set.

The feature list is defined in `config.py` and can be easily modified.

### Models

Two tree-based gradient boosting models are used:

- **XGBoost** (`XGBRegressor`)
- **LightGBM** (`LGBMRegressor`)

Both models are configured to run on GPU (if available) with hand-tuned hyperparameters.

### Training Slices and Time Decay

To capture potential regime changes over time, the training data is split into
three **model slices**:

1. `full_data`: use the entire training history
2. `last_75pct`: use the most recent 75% of the data
3. `last_50pct`: use the most recent 50% of the data

For each slice, a set of **time-decay sample weights** is applied so that
more recent observations receive higher weights during training.

### Cross-Validation and Ensembling

Cross-validation is performed using `KFold` (without shuffling) on the time-ordered data.
For each fold, each base learner (XGBoost / LightGBM) is trained on each data slice,
producing:

- Out-of-fold (OOF) predictions for the training set
- Test predictions, accumulated and averaged over folds

Pearson correlation between OOF predictions and the true labels is used as the main
validation metric.

Ensembling steps:

1. **Within-learner ensemble across slices**  
   - Simple average of slice predictions  
   - Weighted average using slice-level Pearson scores as weights

2. **Across-learner ensemble**  
   - Simple average of the “simple slice ensembles” from XGBoost and LightGBM

The final predictions are written to `submission.csv`.

---

## Repository Structure

```text
.
├─ README.md          # This file
├─ config.py          # Global configuration (paths, features, folds, etc.)
├─ data.py            # Data loading and time-decay weight generation
├─ models.py          # Model definitions and hyperparameters
├─ train.py           # Main training / CV / ensembling script
└─ requirements.txt   # Python dependencies (optional)
