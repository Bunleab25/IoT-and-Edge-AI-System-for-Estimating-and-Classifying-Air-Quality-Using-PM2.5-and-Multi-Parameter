"""Evaluate the three horizon-specific AQI gradient-boosting models."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import _bootstrap  # noqa: F401
from aerosense.evaluation_metrics import (
    align_probabilities,
    classification_metrics,
    hard_prediction_probabilities,
)
from aerosense.experiment_utils import (
    CONFIG_PATH,
    git_revision,
    git_worktree_dirty,
    load_experiment_config,
    package_versions,
    sha256_file,
    source_fingerprints,
    write_json,
)
from aerosense.model_contract import (
    LABEL_MAPPING,
    encode_labels,
    load_model_artifact,
    pm25_to_category,
    prepare_feature_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = PROJECT_ROOT / "data/model_ready/test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUTS_DIR / "evaluation_report.json"
DEMO_REPORT_PATH = OUTPUTS_DIR / "demo_report.json"
CLEANED_CSV = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"
CM_DIR = OUTPUTS_DIR / "confusion_matrices"


def main() -> None:
    """Compare fitted models to chronological persistence and majority baselines."""
    print(f"Loading test data: {TEST_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError("Run 04_train_test_split.py first.")
    train_path = PROJECT_ROOT / "data/model_ready/train.csv"
    if not train_path.exists():
        raise FileNotFoundError("Missing data/model_ready/train.csv; run 04_train_test_split.py first.")
    config = load_experiment_config()
    train = pd.read_csv(train_path)
    test = pd.read_csv(TEST_CSV)
    if train.empty or test.empty or "date" not in train or "date" not in test:
        raise ValueError("Train and test CSV files must contain rows and dates.")
    train_dates = pd.to_datetime(train["date"], errors="raise")
    test_dates = pd.to_datetime(test["date"], errors="raise")
    if not train_dates.is_monotonic_increasing or not test_dates.is_monotonic_increasing:
        raise ValueError("Train and test data must be chronologically ordered.")
    if train_dates.max() >= test_dates.min():
        raise ValueError("Chronological train/test leakage: training dates must end before test dates begin.")
    embargo = config["forecast_horizon_rows"]
    if (test_dates.min() - train_dates.max()).days <= embargo:
        raise ValueError(f"Train/test boundary must embargo at least {embargo} calendar days.")
    test_features = prepare_feature_frame(test)
    train_features = prepare_feature_frame(train)
    reverse_map = {value: label for label, value in LABEL_MAPPING.items()}
    comparison_path = OUTPUTS_DIR / "model_comparison.json"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing {comparison_path}; run 05_train_model.py first.")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if not DEMO_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing {DEMO_REPORT_PATH}; run scripts/run_demo.py first.")
    demo_report = json.loads(DEMO_REPORT_PATH.read_text(encoding="utf-8"))
    expected_ranges = {
        "train_date_range": [str(train["date"].iloc[0]), str(train["date"].iloc[-1])],
        "test_date_range": [str(test["date"].iloc[0]), str(test["date"].iloc[-1])],
    }
    if (
        demo_report.get("train_rows") != len(train)
        or demo_report.get("test_rows") != len(test)
        or demo_report.get("config_sha256") != sha256_file(CONFIG_PATH)
        or demo_report.get("input_sha256") != sha256_file(CLEANED_CSV)
        or any(demo_report.get(key) != value for key, value in expected_ranges.items())
    ):
        raise ValueError(
            "Offline demo results do not match the current training/evaluation holdout. "
            "Rerun scripts/run_demo.py with the current data and experiment config."
        )
    CM_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")
    results = {
        "evaluation": "single chronological holdout",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_date_range": [str(train["date"].iloc[0]), str(train["date"].iloc[-1])],
        "test_date_range": [str(test["date"].iloc[0]), str(test["date"].iloc[-1])],
        "purge_gap_rows": embargo,
        "config_sha256": sha256_file(CONFIG_PATH),
        "train_csv_sha256": sha256_file(train_path),
        "test_csv_sha256": sha256_file(TEST_CSV),
        "git_revision": git_revision(),
        "git_worktree_dirty": git_worktree_dirty(),
        "source_sha256": source_fingerprints(),
        "packages": package_versions(),
        "search_results": comparison,
        "horizons": {},
    }

    for horizon in range(1, 4):
        target = f"label_day{horizon}"
        if target not in train or target not in test or "aqi_category" not in test:
            raise ValueError(f"Missing target or persistence label data for horizon {horizon}.")
        model_paths = {
            "xgboost": MODELS_DIR / f"xgb_day{horizon}.pkl",
            "lightgbm": MODELS_DIR / f"lgbm_day{horizon}.pkl",
            "selected": MODELS_DIR / f"best_day{horizon}.pkl",
        }
        missing_models = [str(path) for path in model_paths.values() if not path.exists()]
        if missing_models:
            raise FileNotFoundError(
                f"Missing model artifacts {missing_models}; run scripts/05_train_model.py first."
            )
        artifacts = {name: load_model_artifact(path) for name, path in model_paths.items()}
        selected_model_name = comparison.get(f"day{horizon}", {}).get("selected")
        if selected_model_name not in {"xgboost", "lightgbm"}:
            raise ValueError(f"Missing or invalid selected model for day{horizon} in {comparison_path}.")

        y_train = encode_labels(train[target]).to_numpy()
        actual = encode_labels(test[target])
        persistence_predictions = encode_labels(test["aqi_category"]).to_numpy()
        majority_class = int(np.bincount(y_train, minlength=len(LABEL_MAPPING)).argmax())
        baselines = {
            "persistence": persistence_predictions,
            "training_majority": np.full(len(test), majority_class, dtype=int),
        }

        model_results = {}
        predictions_by_name = {}
        probability_by_name = {}
        for name in ("xgboost", "lightgbm"):
            estimator = artifacts[name]["estimator"]
            prediction = estimator.predict(test_features).astype(int)
            probabilities = align_probabilities(estimator.predict_proba(test_features), estimator.classes_)
            predictions_by_name[name] = prediction
            probability_by_name[name] = probabilities
            model_results[name] = classification_metrics(actual, prediction, probabilities)

        model = artifacts["selected"]["estimator"]
        selected_predictions = model.predict(test_features).astype(int)
        selected_probabilities = align_probabilities(model.predict_proba(test_features), model.classes_)
        predictions_by_name["selected_model"] = selected_predictions
        probability_by_name["selected_model"] = selected_probabilities
        selected_metrics = classification_metrics(actual, selected_predictions, selected_probabilities)

        demo_model = RandomForestClassifier(
            n_estimators=config["demo"]["n_estimators"],
            random_state=config["seed"],
            n_jobs=1,
            class_weight="balanced",
        )
        demo_model.fit(train_features, y_train)
        demo_predictions = demo_model.predict(test_features)
        demo_probabilities = align_probabilities(demo_model.predict_proba(test_features), demo_model.classes_)
        demo_metrics = classification_metrics(actual, demo_predictions, demo_probabilities)
        demo_metrics_reported = demo_report["horizons"][f"day{horizon}"]["model_metrics"]
        if demo_metrics != demo_metrics_reported:
            raise ValueError(
                f"Offline demo metrics for day{horizon} do not reproduce on the current holdout."
            )
        predictions_by_name["offline_demo_random_forest"] = demo_predictions
        probability_by_name["offline_demo_random_forest"] = demo_probabilities
        model_results["offline_demo_random_forest"] = demo_metrics

        present_labels = sorted(
            set(actual).union(
                *(set(values) for values in predictions_by_name.values()),
                {majority_class},
            )
        )
        label_names = [reverse_map.get(label, str(label)) for label in present_labels]
        baseline_results = {}
        predictions_by_name.update(baselines)
        probability_by_name.update({
            name: hard_prediction_probabilities(values)
            for name, values in baselines.items()
        })
        for name, baseline_predictions in baselines.items():
            baseline_results[name] = classification_metrics(
                actual,
                baseline_predictions,
                probability_by_name[name],
            )

        matrix_files = {}
        for name, predicted in predictions_by_name.items():
            matrix = confusion_matrix(actual, predicted, labels=present_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_names,
                yticklabels=label_names,
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix — Day +{horizon}")
            plt.tight_layout()
            cm_path = CM_DIR / f"day{horizon}_{name}_confusion_matrix.png"
            plt.savefig(cm_path, dpi=160)
            plt.close()
            matrix_files[name] = str(cm_path.relative_to(PROJECT_ROOT))

        importances = dict(zip(
            artifacts["selected"]["feature_names"],
            (float(value) for value in model.feature_importances_),
        ))
        results["horizons"][f"day{horizon}"] = {
            "selected_model_name": selected_model_name,
            "selected_model": selected_metrics,
            "models": model_results,
            "baselines": baseline_results,
            "confusion_matrix_files": matrix_files,
            "feature_importances": importances,
        }
        print(
            f"Day +{horizon}: XGBoost={model_results['xgboost']['macro_f1']:.3f}, "
            f"LightGBM={model_results['lightgbm']['macro_f1']:.3f}, "
            f"demo RF={demo_metrics['macro_f1']:.3f}, "
            f"persistence={baseline_results['persistence']['macro_f1']:.3f}, "
            f"majority={baseline_results['training_majority']['macro_f1']:.3f}"
        )

    results["macro_f1_degradation_day1_to_day3"] = float(
        results["horizons"]["day1"]["selected_model"]["macro_f1"]
        - results["horizons"]["day3"]["selected_model"]["macro_f1"]
    )
    write_json(REPORT_PATH, results)
    print(f"Saved evaluation report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
