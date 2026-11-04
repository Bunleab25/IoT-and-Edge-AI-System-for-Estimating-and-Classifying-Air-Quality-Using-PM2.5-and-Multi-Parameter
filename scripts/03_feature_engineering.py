"""Build horizon-specific future AQI labels from cleaned daily observations."""
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from aerosense.model_contract import (
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    add_future_labels,
    engineer_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/model_ready/train_test_shifted.csv"
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "outputs/feature_contract.json"


def main() -> None:
    """Add lag/rolling/trend/seasonal features and shift AQI labels one to three rows ahead."""
    print(f"Loading cleaned data: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError("Run 01_data_cleaning.py first.")

    raw = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = add_future_labels(engineer_features(raw))
    print(f"Built {len(FEATURE_COLUMNS)} causal features using the shared model contract.")

    lead_nan_cols = [
        "pm25_lag1", "pm25_lag2", "pm25_lag3",
        "temp_lag1", "temp_lag2", "temp_lag3",
        "humidity_lag1", "humidity_lag2", "humidity_lag3",
        "windspeed_lag1", "windspeed_lag2", "windspeed_lag3",
        "pm25_trend3", "pressure_lag1", "pressure_trend3",
        "aqi_category_lag1",
    ]
    rows_before_drop = len(df)
    df = df.dropna(subset=lead_nan_cols).copy()
    print(f"Dropped {rows_before_drop - len(df)} leading rows with NaN lag/trend features.")

    original_rows = len(df)
    df = df.dropna(subset=["label_day1", "label_day2", "label_day3"]).copy()
    print(f"Dropped {original_rows - len(df)} final row(s) without all future targets.")

    print(f"Final feature columns: {FEATURE_COLUMNS}")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    FEATURE_CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_CONTRACT_PATH.write_text(
        json.dumps({
            "version": FEATURE_CONTRACT_VERSION,
            "feature_names": FEATURE_COLUMNS,
            "stagnation_normalization": "causal_trailing_30_observations",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(df)} model-ready rows to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
