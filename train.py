# train.py
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold  # 可按需要换成 TimeSeriesSplit

from config import DEFAULT_CONFIG
from data import load_data, create_time_decay_weights
from models import get_learners


def main() -> None:
    config = DEFAULT_CONFIG

    # Load data
    train_df, test_df, submission_df = load_data(config)
    n_samples = len(train_df)

    # Define model slices (full history vs. recent 75% / 50%)
    model_slices = [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples)},
    ]

    learners = get_learners(random_state=config.random_state)

    # Storage for OOF and test predictions
    oof_preds: Dict[str, Dict[str, np.ndarray]] = {
        learner["name"]: {sl["name"]: np.zeros(n_samples, dtype=float) for sl in model_slices}
        for learner in learners
    }
    test_preds: Dict[str, Dict[str, np.ndarray]] = {
        learner["name"]: {sl["name"]: np.zeros(len(test_df), dtype=float) for sl in model_slices}
        for learner in learners
    }

    # Global time-decay weights for full sample
    full_weights = create_time_decay_weights(n_samples)

    # NOTE: for strict time-series CV, you can switch to TimeSeriesSplit
    # from sklearn.model_selection import TimeSeriesSplit
    # cv = TimeSeriesSplit(n_splits=config.n_folds)
    cv = KFold(n_splits=config.n_folds, shuffle=False)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{config.n_folds} ---")

        X_valid = train_df.iloc[valid_idx][config.features]
        y_valid = train_df.iloc[valid_idx][config.label_column]

        for sl in model_slices:
            slice_name = sl["name"]
            cutoff = sl["cutoff"]

            # Use only data after cutoff for this slice
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff

            X_train = subset.iloc[rel_idx][config.features]
            y_train = subset.iloc[rel_idx][config.label_column]

            # Sample weights (time-decay)
            if cutoff == 0:
                sw = full_weights[train_idx]
            else:
                sw_total = create_time_decay_weights(len(subset))
                sw = sw_total[rel_idx]

            for learner in learners:
                name = learner["name"]
                Estimator = learner["Estimator"]
                params = learner["params"]

                model = Estimator(**params)

                # Plain fit; you can add eval_set & early_stopping if desired
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=sw,
                )

                # Out-of-fold predictions
                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[name][slice_name][idxs] = model.predict(
                        train_df.iloc[idxs][config.features]
                    )
                # For earlier indices (before cutoff), fall back to full-data model
                if cutoff > 0 and (~mask).any():
                    oof_preds[name][slice_name][valid_idx[~mask]] = (
                        oof_preds[name]["full_data"][valid_idx[~mask]]
                    )

                # Test predictions (accumulate over folds)
                test_preds[name][slice_name] += model.predict(test_df[config.features])

    # Average test preds over folds
    for name in test_preds:
        for slice_name in test_preds[name]:
            test_preds[name][slice_name] /= config.n_folds

    # Pearson scores per learner and slice
    pearson_scores = {
        name: {
            slice_name: pearsonr(train_df[config.label_column], preds)[0]
            for slice_name, preds in slices.items()
        }
        for name, slices in oof_preds.items()
    }

    print("\nPearson scores by learner and slice:")
    for learner_name, scores in pearson_scores.items():
        print(learner_name, scores)

    # Ensemble per learner across slices
    learner_ensembles: Dict[str, Dict[str, np.ndarray]] = {}
    for learner_name, slice_scores in pearson_scores.items():
        # simple average across slices
        oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
        test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
        score_simple = pearsonr(train_df[config.label_column], oof_simple)[0]

        # weighted ensemble across slices using Pearson scores as weights
        total_score = sum(slice_scores.values())
        slice_weights = {sn: sc / total_score for sn, sc in slice_scores.items()}
        oof_weighted = sum(
            slice_weights[sn] * oof_preds[learner_name][sn] for sn in slice_weights
        )
        test_weighted = sum(
            slice_weights[sn] * test_preds[learner_name][sn] for sn in slice_weights
        )
        score_weighted = pearsonr(train_df[config.label_column], oof_weighted)[0]

        print(f"\n{learner_name.upper()} simple ensemble Pearson:   {score_simple:.4f}")
        print(f"{learner_name.upper()} weighted ensemble Pearson: {score_weighted:.4f}")

        learner_ensembles[learner_name] = {
            "oof_simple": oof_simple,
            "test_simple": test_simple,
        }

    # Final ensemble across learners
    final_oof = np.mean(
        [le["oof_simple"] for le in learner_ensembles.values()], axis=0
    )
    final_test = np.mean(
        [le["test_simple"] for le in learner_ensembles.values()], axis=0
    )
    final_score = pearsonr(train_df[config.label_column], final_oof)[0]
    print(f"\nFINAL ensemble across learners Pearson: {final_score:.4f}")

    # Save submission
    submission_df["prediction"] = final_test
    submission_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
