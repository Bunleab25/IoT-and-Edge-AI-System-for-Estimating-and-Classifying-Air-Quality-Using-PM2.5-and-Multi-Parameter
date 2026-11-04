"""Run a deterministic, offline end-to-end forecast demo from checked-in data."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401,E402
from aerosense.evaluation_metrics import align_probabilities, classification_metrics  # noqa: E402
from aerosense.experiment_utils import (  # noqa: E402
    CONFIG_PATH,
    git_revision,
    git_worktree_dirty,
    load_experiment_config,
    package_versions,
    sha256_file,
    source_fingerprints,
    write_json,
)
from aerosense.model_contract import (  # noqa: E402
    FEATURE_COLUMNS,
    LABEL_MAPPING,
    add_future_labels,
    engineer_features,
    encode_labels,
    pm25_to_category,
    prepare_feature_frame,
)

CLEANED_CSV = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/demo_report.json"


def build_labeled_rows(cleaned: pd.DataFrame, horizon_rows: int) -> pd.DataFrame:
    """Create causal features and future categorical targets for the demo."""
    data = add_future_labels(engineer_features(cleaned), horizon_rows)
    required = list(FEATURE_COLUMNS) + [f"label_day{horizon}" for horizon in range(1, horizon_rows + 1)]
    return data.dropna(subset=required).reset_index(drop=True)


def run_demo() -> dict:
    """Fit small random forests per horizon on a purged earlier-data split."""
    config = load_experiment_config()
    demo_config = config.get("demo", {})
    estimator_count = demo_config.get("n_estimators")
    if not isinstance(estimator_count, int) or estimator_count < 1:
        raise ValueError("Demo n_estimators must be a positive integer.")
    if not CLEANED_CSV.exists():
        raise FileNotFoundError(f"Missing {CLEANED_CSV}; run scripts/01_data_cleaning.py first.")

    cleaned = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    data = build_labeled_rows(cleaned, config["forecast_horizon_rows"])
    split_index = int(len(data) * config["train_fraction"])
    purge = config["forecast_horizon_rows"]
    train = data.iloc[:split_index - purge].copy()
    test = data.iloc[split_index:].copy()
    if train.empty or test.empty:
        raise ValueError("Dataset is too small for the configured demo split and purge gap.")
    if pd.to_datetime(train["date"]).max() >= pd.to_datetime(test["date"]).min():
        raise ValueError("Demo split is not chronological.")
    if (
        pd.to_datetime(test["date"]).min() - pd.to_datetime(train["date"]).max()
    ).days <= purge:
        raise ValueError(f"Demo split must embargo at least {purge} calendar days.")

    X_train = prepare_feature_frame(train)
    X_test = prepare_feature_frame(test)
    results = {
        "demo_only": True,
        "model": "RandomForestClassifier",
        "hyperparameter_search": False,
        "seed": config["seed"],
        "train_rows": len(train),
        "test_rows": len(test),
        "purge_gap_rows": purge,
        "train_date_range": [str(train["date"].iloc[0].date()), str(train["date"].iloc[-1].date())],
        "test_date_range": [str(test["date"].iloc[0].date()), str(test["date"].iloc[-1].date())],
        "input_sha256": sha256_file(CLEANED_CSV),
        "config_sha256": sha256_file(CONFIG_PATH),
        "git_revision": git_revision(),
        "git_worktree_dirty": git_worktree_dirty(),
        "source_sha256": source_fingerprints(),
        "packages": package_versions(),
        "horizons": {},
    }

    for horizon in range(1, config["forecast_horizon_rows"] + 1):
        target = f"label_day{horizon}"
        y_train = encode_labels(train[target]).to_numpy()
        y_test = encode_labels(test[target]).to_numpy()
        estimator = RandomForestClassifier(
            n_estimators=estimator_count,
            random_state=config["seed"],
            n_jobs=1,
            class_weight="balanced",
        )
        estimator.fit(X_train, y_train)
        prediction = estimator.predict(X_test)
        probabilities = align_probabilities(estimator.predict_proba(X_test), estimator.classes_)
        persistence = encode_labels(test["pm25"].map(pm25_to_category)).to_numpy()
        results["horizons"][f"day{horizon}"] = {
            "model_metrics": classification_metrics(y_test, prediction, probabilities),
            "persistence_baseline_metrics": classification_metrics(
                y_test,
                persistence,
                align_probabilities(
                    np.eye(len(LABEL_MAPPING))[persistence],
                    np.arange(len(LABEL_MAPPING)),
                ),
            ),
        }

    write_json(REPORT_PATH, results)
    return results


if __name__ == "__main__":
    report = run_demo()
    print(f"Offline demo complete: {report['train_rows']} train / {report['test_rows']} test rows.")
    for horizon, result in report["horizons"].items():
        print(
            f"{horizon}: demo macro-F1={result['model_metrics']['macro_f1']:.3f}; "
            f"persistence macro-F1={result['persistence_baseline_metrics']['macro_f1']:.3f}"
        )
    print(f"Detailed demo report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
