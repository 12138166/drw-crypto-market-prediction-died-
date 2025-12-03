# config.py
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Config:
    """
    Global configuration for the DRW - Crypto Market Prediction project.
    Update the paths below according to your local environment or Kaggle paths.
    """
    train_path: Path = Path("/kaggle/input/drw-crypto-market-prediction/train.parquet")
    test_path: Path = Path("/kaggle/input/drw-crypto-market-prediction/test.parquet")
    submission_path: Path = Path("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")

    # Features used for training
    features: List[str] = None
    label_column: str = "label"
    n_folds: int = 3
    random_state: int = 42

    def __post_init__(self):
        if self.features is None:
            self.features = [
                "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603",
                "X860", "X674", "X415", "X345", "X137", "X855", "X174", "X302",
                "X178", "X532", "X168", "X612",
                "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
            ]


DEFAULT_CONFIG = Config()
