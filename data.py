# data.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from config import Config, DEFAULT_CONFIG


def create_time_decay_weights(n: int, decay: float = 0.95) -> np.ndarray:
    """
    Create time-decay sample weights (more weight on recent observations).

    Parameters
    ----------
    n : int
        Number of observations.
    decay : float
        Decay factor in (0, 1]. Smaller means more aggressive decay.

    Returns
    -------
    np.ndarray
        Array of length n with weights summing approximately to n.
    """
    if n <= 1:
        return np.ones(n, dtype=float)

    positions = np.arange(n)
    normalized = positions / float(n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()


def load_data(config: Config = DEFAULT_CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, test, and sample submission dataframes.

    Parameters
    ----------
    config : Config
        Project configuration with paths and feature list.

    Returns
    -------
    (train_df, test_df, submission_df)
    """
    train_df = pd.read_parquet(
        config.train_path,
        columns=config.features + [config.label_column],
    ).reset_index(drop=True)

    test_df = pd.read_parquet(
        config.test_path,
        columns=config.features,
    ).reset_index(drop=True)

    submission_df = pd.read_csv(config.submission_path)

    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}, submission: {submission_df.shape}")
    return train_df, test_df, submission_df
