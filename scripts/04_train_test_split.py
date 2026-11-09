"""Create a chronological train/test split for AQI forecasting."""
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from aerosense.experiment_utils import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data/model_ready/train_test_shifted.csv"
TRAIN_CSV = PROJECT_ROOT / "data/model_ready/train.csv"
TEST_CSV = PROJECT_ROOT / "data/model_ready/test.csv"


def chronological_purged_split(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split in time order and purge rows whose future labels overlap the holdout."""
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("At least 10 rows are required for a useful time-based split.")
    split_index = int(len(df) * config["train_fraction"])
    purge_rows = config["forecast_horizon_rows"]
    if split_index <= purge_rows + config["time_series_cv_splits"] + 1 or split_index >= len(df):
        raise ValueError("Dataset is too small for the configured chronological split and CV folds.")
    train = df.iloc[:split_index - purge_rows].copy()
    test = df.iloc[split_index:].copy()
    if pd.to_datetime(train["date"]).max() >= pd.to_datetime(test["date"]).min():
        raise ValueError("Chronological split did not separate train and test dates.")
    return train, test


def main() -> None:
    """Split in chronological order using the configured training fraction."""
    config = load_experiment_config()
    test_fraction = 1 - config["train_fraction"]
    print(f"Loading model-ready data: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError("Run 03_feature_engineering.py first.")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    purge_rows = config["forecast_horizon_rows"]
    train, test = chronological_purged_split(df, config)
    for frame in (train, test):
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"Purged time-based split (~{config['train_fraction']:.0%}/{test_fraction:.0%} before purge, "
          f"{purge_rows} target rows embargoed): {len(train)} train rows through {train['date'].iloc[-1]}; "
          f"{len(test)} test rows from {test['date'].iloc[0]}.")
    print(f"Saved: {TRAIN_CSV} and {TEST_CSV}")


if __name__ == "__main__":
    main()
