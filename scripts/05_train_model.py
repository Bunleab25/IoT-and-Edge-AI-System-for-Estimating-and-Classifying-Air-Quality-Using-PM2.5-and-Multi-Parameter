"""Train and compare XGBoost and LightGBM models per AQI forecast horizon."""
import json
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

import _bootstrap  # noqa: F401
from aerosense.experiment_utils import (
    CONFIG_PATH,
    build_training_manifest,
    load_experiment_config,
    write_json,
)
from aerosense.model_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_COLUMNS,
    LABEL_MAPPING,
    encode_labels,
    prepare_feature_frame,
    save_model_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "data/model_ready/train.csv"
TEST_CSV = PROJECT_ROOT / "data/model_ready/test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LABEL_ENCODING_PATH = OUTPUTS_DIR / "label_encoding.json"
COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.json"
FEATURE_CONTRACT_PATH = OUTPUTS_DIR / "feature_contract.json"
TRAINING_MANIFEST_PATH = OUTPUTS_DIR / "training_run.json"


def _load_train_and_test(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load chronological train and test sets."""
    if not FEATURE_CONTRACT_PATH.exists():
        raise FileNotFoundError("Run 03_feature_engineering.py and 04_train_test_split.py first.")
    feature_contract = json.loads(FEATURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    if (
        feature_contract.get("version") != FEATURE_CONTRACT_VERSION
        or feature_contract.get("feature_names") != FEATURE_COLUMNS
    ):
        raise ValueError(
            "The prepared dataset uses an incompatible feature contract; rerun "
            "03_feature_engineering.py and 04_train_test_split.py."
        )
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError("Run 04_train_test_split.py first.")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    if "date" not in train or "date" not in test:
        raise ValueError("Train and test CSV files must contain dates for chronological validation.")
    train_dates = pd.to_datetime(train["date"], errors="raise")
    test_dates = pd.to_datetime(test["date"], errors="raise")
    if not train_dates.is_monotonic_increasing or not test_dates.is_monotonic_increasing:
        raise ValueError("Train and test data must already be in chronological order.")
    if train_dates.max() >= test_dates.min():
        raise ValueError("Chronological train/test leakage: training dates must end before test dates begin.")
    if (test_dates.min() - train_dates.max()).days <= config["forecast_horizon_rows"]:
        raise ValueError(
            f"Train/test boundary must embargo at least {config['forecast_horizon_rows']} target rows."
        )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    minimum_train_rows = config["time_series_cv_splits"] + config["forecast_horizon_rows"] + 1
    if len(train) < minimum_train_rows:
        raise ValueError(
            f"At least {minimum_train_rows} training rows are required for "
            f"{config['time_series_cv_splits']} chronological CV splits."
        )
    return train, test


def _save_label_mapping() -> None:
    """Persist the fixed mapping used to convert category names to ordinal integers."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_ENCODING_PATH.write_text(json.dumps(LABEL_MAPPING, indent=2), encoding="utf-8")
    print(f"Saved label encoding: {LABEL_ENCODING_PATH}")


def _save_feature_importance(path: Path, feature_names: list[str], importances: list[float], model_name: str) -> None:
    """Write model feature importances to JSON for inspection."""
    payload = {
        "model": model_name,
        "feature_names": feature_names,
        "importances": [float(value) for value in importances],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved feature importance: {path}")


def _fit_xgb(X_train: pd.DataFrame, y_train: pd.Series, config: dict) -> RandomizedSearchCV:
    """Tune a time-series XGBoost classifier with a macro-F1 objective."""
    xgb_model = XGBClassifier(
        objective="multi:softmax",
        num_class=4,
        eval_metric="mlogloss",
        random_state=config["seed"],
        n_jobs=1,
        verbosity=0,
    )
    tscv = TimeSeriesSplit(
        n_splits=config["time_series_cv_splits"],
        gap=config["forecast_horizon_rows"],
    )
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=config["xgboost"]["parameters"],
        n_iter=config["search_iterations"],
        cv=tscv,
        scoring=config["selection_metric"],
        random_state=config["seed"],
        n_jobs=config.get("search_n_jobs", 1),
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def _fit_lgbm(X_train: pd.DataFrame, y_train: pd.Series, config: dict) -> RandomizedSearchCV:
    """Tune a time-series LightGBM classifier with a macro-F1 objective."""
    lgbm_model = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        random_state=config["seed"],
        n_jobs=1,
        verbose=-1,
    )
    tscv = TimeSeriesSplit(
        n_splits=config["time_series_cv_splits"],
        gap=config["forecast_horizon_rows"],
    )
    search = RandomizedSearchCV(
        estimator=lgbm_model,
        param_distributions=config["lightgbm"]["parameters"],
        n_iter=config["search_iterations"],
        cv=tscv,
        scoring=config["selection_metric"],
        random_state=config["seed"],
        n_jobs=config.get("search_n_jobs", 1),
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def main() -> None:
    """Train XGBoost and LightGBM per horizon, select the better model, and save artifacts."""
    start_time = time.perf_counter()
    config = load_experiment_config()
    train, test = _load_train_and_test(config)
    _save_label_mapping()
    X_train = prepare_feature_frame(train)
    prepare_feature_frame(test)

    print(f"Using feature columns: {FEATURE_COLUMNS}")
    print(f"Training rows: {len(train)} | test rows: {len(test)}")
    print(
        "Purged chronological cross-validation: "
        f"TimeSeriesSplit(n_splits={config['time_series_cv_splits']}, "
        f"gap={config['forecast_horizon_rows']}); seed={config['seed']}; "
        f"metric={config['selection_metric']}"
    )

    comparison = {}
    selected_models = {}
    for horizon in range(1, 4):
        target_name = f"label_day{horizon}"
        if target_name not in train.columns:
            raise ValueError(f"Missing target column {target_name} in training data.")

        y_train = encode_labels(train[target_name])

        print(f"\n=== Horizon {horizon}: {target_name} ===")
        print("Training XGBoost...")
        xgb_search = _fit_xgb(X_train, y_train, config)
        xgb_best_score = float(xgb_search.best_score_)
        print(f"XGBoost best CV F1 (macro): {xgb_best_score:.4f}")

        print("Training LightGBM...")
        lgbm_search = _fit_lgbm(X_train, y_train, config)
        lgbm_best_score = float(lgbm_search.best_score_)
        print(f"LightGBM best CV F1 (macro): {lgbm_best_score:.4f}")

        selected_model_name = "xgboost" if xgb_best_score >= lgbm_best_score else "lightgbm"
        selected_model = xgb_search.best_estimator_ if selected_model_name == "xgboost" else lgbm_search.best_estimator_
        print(f"Selected model for {target_name}: {selected_model_name}")

        xgb_path = MODELS_DIR / f"xgb_day{horizon}.pkl"
        lgbm_path = MODELS_DIR / f"lgbm_day{horizon}.pkl"
        best_path = MODELS_DIR / f"best_day{horizon}.pkl"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        save_model_artifact(xgb_search.best_estimator_, xgb_path, horizon)
        save_model_artifact(lgbm_search.best_estimator_, lgbm_path, horizon)
        save_model_artifact(selected_model, best_path, horizon)
        print(f"Saved: {xgb_path}, {lgbm_path}, {best_path}")

        feature_importance_path = OUTPUTS_DIR / f"feature_importance_day{horizon}.json"
        feature_importance = selected_model.feature_importances_
        _save_feature_importance(feature_importance_path, FEATURE_COLUMNS, feature_importance, selected_model_name)

        comparison[f"day{horizon}"] = {
            "xgboost_cv_f1": round(xgb_best_score, 6),
            "xgboost_cv_f1_std": float(xgb_search.cv_results_["std_test_score"][xgb_search.best_index_]),
            "lightgbm_cv_f1": round(lgbm_best_score, 6),
            "lightgbm_cv_f1_std": float(lgbm_search.cv_results_["std_test_score"][lgbm_search.best_index_]),
            "selected": selected_model_name,
            "xgboost_best_params": xgb_search.best_params_,
            "lightgbm_best_params": lgbm_search.best_params_,
        }
        selected_models[f"day{horizon}"] = {
            "name": selected_model_name,
            "cv_f1_macro": max(xgb_best_score, lgbm_best_score),
            "best_params": (
                xgb_search.best_params_ if selected_model_name == "xgboost"
                else lgbm_search.best_params_
            ),
        }

    write_json(COMPARISON_PATH, comparison)
    print(f"Saved model comparison summary: {COMPARISON_PATH}")

    manifest = build_training_manifest(
        config=config,
        config_path=CONFIG_PATH,
        train_path=TRAIN_CSV,
        test_path=TEST_CSV,
        feature_contract_path=FEATURE_CONTRACT_PATH,
        split_summary={
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": str(train["date"].iloc[0]),
            "train_end": str(train["date"].iloc[-1]),
            "test_start": str(test["date"].iloc[0]),
            "test_end": str(test["date"].iloc[-1]),
            "test_fraction": len(test) / (len(train) + len(test)),
        },
        selected_models=selected_models,
    )
    manifest["model_comparison"] = comparison
    write_json(TRAINING_MANIFEST_PATH, manifest)
    print(f"Saved training provenance: {TRAINING_MANIFEST_PATH}")

    runtime_seconds = time.perf_counter() - start_time
    print(f"Total training runtime: {runtime_seconds:.2f} seconds for 2 models x 3 horizons (6 randomized searches).")


if __name__ == "__main__":
    main()
