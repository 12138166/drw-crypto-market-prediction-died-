# models.py
from __future__ import annotations

from typing import Any, Dict, List

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def get_learners(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Define the base learners (XGBoost and LightGBM) and their hyperparameters.
    """
    xgb_params = {
        "tree_method": "hist",
        "device": "gpu",
        "colsample_bylevel": 0.4778,
        "colsample_bynode": 0.3628,
        "colsample_bytree": 0.7107,
        "gamma": 1.7095,
        "learning_rate": 0.02213,
        "max_depth": 20,
        "max_leaves": 12,
        "min_child_weight": 16,
        "n_estimators": 1667,
        "subsample": 0.06567,
        "reg_alpha": 39.3524,
        "reg_lambda": 75.4484,
        "verbosity": 0,
        "random_state": random_state,
        "n_jobs": -1,
    }

    lgbm_params = {
        "boosting_type": "gbdt",
        "device": "gpu",
        "n_jobs": -1,
        "verbose": -1,
        "random_state": random_state,
        "colsample_bytree": 0.5039,
        "learning_rate": 0.01260,
        "min_child_samples": 20,
        "min_child_weight": 0.1146,
        "n_estimators": 915,
        "num_leaves": 145,
        "reg_alpha": 19.2447,
        "reg_lambda": 55.5046,
        "subsample": 0.9709,
        "max_depth": 9,
    }

    learners = [
        {"name": "xgb", "Estimator": XGBRegressor, "params": xgb_params},
        {"name": "lgbm", "Estimator": LGBMRegressor, "params": lgbm_params},
    ]
    return learners
