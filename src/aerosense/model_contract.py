"""Shared feature and model-artifact contract for training and inference."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TARGET_LABELS = ["Very Good", "Good", "Moderate", "Unhealthy"]
LABEL_MAPPING = {label: index for index, label in enumerate(TARGET_LABELS)}
MODEL_ARTIFACT_FORMAT = "aerosense-aqi-model"
MODEL_ARTIFACT_VERSION = 1
FEATURE_CONTRACT_VERSION = 1
OBSERVATION_COLUMNS = [
    "pm25",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "windspeed_10m",
    "winddirection_10m",
]
FEATURE_COLUMNS = [
    "pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "windspeed_10m", "winddirection_10m",
    "pm25_lag1", "pm25_lag2", "pm25_lag3",
    "pm25_roll_mean3", "pm25_roll_mean7", "pm25_trend3",
    "temp_lag1", "temp_lag2", "temp_lag3",
    "temp_roll_mean3", "temp_roll_mean7",
    "humidity_lag1", "humidity_lag2", "humidity_lag3",
    "humidity_roll_mean3", "humidity_roll_mean7",
    "pressure_lag1", "pressure_trend3",
    "windspeed_lag1", "windspeed_lag2", "windspeed_lag3",
    "windspeed_roll_mean3", "windspeed_roll_mean7",
    "wind_from_burning_sector", "wind_direction_sin", "wind_direction_cos",
    "aqi_category_lag1", "day_of_year", "month", "month_sin", "month_cos",
    "stagnation_score",
]


def pm25_to_category(pm25: float) -> str:
    """Map PM2.5 to the project's four AQI categories."""
    if pm25 <= 15:
        return "Very Good"
    if pm25 <= 25:
        return "Good"
    if pm25 <= 37.5:
        return "Moderate"
    return "Unhealthy"


def encode_labels(series: pd.Series) -> pd.Series:
    """Encode known AQI labels and reject values outside the model contract."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="raise")
        if (numeric.dropna() % 1 != 0).any():
            raise ValueError("Numeric AQI labels must be integer class ids.")
        encoded = numeric.astype("Int64")
    else:
        encoded = series.map(LABEL_MAPPING)
    invalid = encoded.isna() | ~encoded.isin(range(len(TARGET_LABELS)))
    if invalid.any():
        values = series[invalid].drop_duplicates().astype(str).tolist()
        raise ValueError(f"Unknown AQI label(s): {values}")
    return encoded.astype(int)


def _causal_zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """Normalize each value against up to 30 preceding observations only."""
    previous = series.shift(1)
    mean = previous.rolling(window=window, min_periods=2).mean()
    std = previous.rolling(window=window, min_periods=2).std(ddof=0)
    score = (series - mean) / std
    return score.where(std.notna() & std.ne(0), 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def engineer_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Build the ordered training/inference feature schema from chronological observations."""
    required = {"date", *OBSERVATION_COLUMNS}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Missing observation column(s): {missing}")

    data = observations.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("Observation dates must be valid.")
    if data["date"].duplicated().any():
        raise ValueError("Observation dates must be unique.")
    for column in OBSERVATION_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[OBSERVATION_COLUMNS].isna().any().any():
        raise ValueError("Observation values must not be missing.")
    data["winddirection_10m"] = data["winddirection_10m"] % 360
    data = data.sort_values("date").reset_index(drop=True)

    for lag in (1, 2, 3):
        data[f"pm25_lag{lag}"] = data["pm25"].shift(lag)
        data[f"temp_lag{lag}"] = data["temperature_2m"].shift(lag)
        data[f"humidity_lag{lag}"] = data["relative_humidity_2m"].shift(lag)
        data[f"windspeed_lag{lag}"] = data["windspeed_10m"].shift(lag)
    for window in (3, 7):
        data[f"pm25_roll_mean{window}"] = data["pm25"].rolling(window=window, min_periods=1).mean()
        data[f"temp_roll_mean{window}"] = data["temperature_2m"].rolling(window=window, min_periods=1).mean()
        data[f"humidity_roll_mean{window}"] = data["relative_humidity_2m"].rolling(window=window, min_periods=1).mean()
        data[f"windspeed_roll_mean{window}"] = data["windspeed_10m"].rolling(window=window, min_periods=1).mean()

    data["pm25_trend3"] = data["pm25"] - data["pm25"].shift(3)
    data["pressure_lag1"] = data["surface_pressure"].shift(1)
    data["pressure_trend3"] = data["surface_pressure"] - data["surface_pressure"].shift(3)
    data["wind_from_burning_sector"] = data["winddirection_10m"].between(0, 90).astype(int)
    data["wind_direction_sin"] = np.sin(np.deg2rad(data["winddirection_10m"]))
    data["wind_direction_cos"] = np.cos(np.deg2rad(data["winddirection_10m"]))
    data["aqi_category_lag1"] = data["pm25"].shift(1).map(pm25_to_category).map(LABEL_MAPPING)
    data["day_of_year"] = data["date"].dt.day_of_year
    data["month"] = data["date"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    low_wind = -_causal_zscore(data["windspeed_roll_mean3"])
    high_pressure = _causal_zscore(data["surface_pressure"])
    low_temp_trend = -_causal_zscore(data["temperature_2m"] - data["temperature_2m"].shift(3))
    data["stagnation_score"] = (low_wind + high_pressure + low_temp_trend) / 3.0
    return data


def add_future_labels(frame: pd.DataFrame, horizons: int = 3) -> pd.DataFrame:
    """Attach labels from exact future calendar dates rather than adjacent rows."""
    if horizons < 1:
        raise ValueError("At least one forecast horizon is required.")
    if {"date", "pm25"}.difference(frame.columns):
        raise ValueError("Future labels require date and pm25 columns.")
    data = frame.copy()
    dates = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    if dates.duplicated().any():
        raise ValueError("Future labels require at most one observation per calendar date.")
    categories_by_date = pd.Series(
        data["pm25"].map(pm25_to_category).to_numpy(),
        index=dates,
    )
    for horizon in range(1, horizons + 1):
        data[f"label_day{horizon}"] = (dates + pd.Timedelta(days=horizon)).map(categories_by_date)
    return data


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and order a frame exactly as required by the trained estimators."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing model feature(s): {missing}")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in features.columns:
        if column == "aqi_category_lag1" and not pd.api.types.is_numeric_dtype(features[column]):
            features[column] = encode_labels(features[column])
        features[column] = pd.to_numeric(features[column], errors="raise")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Model features must contain only finite numeric values.")
    return features


def make_model_artifact(estimator, horizon: int) -> dict:
    """Wrap a selected estimator with the exact schema and label mapping it was trained on."""
    return {
        "format": MODEL_ARTIFACT_FORMAT,
        "version": MODEL_ARTIFACT_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "horizon": horizon,
        "estimator": estimator,
        "feature_names": list(FEATURE_COLUMNS),
        "label_mapping": dict(LABEL_MAPPING),
    }


def _validate_estimator_schema(estimator, feature_names: list[str], path: Path) -> None:
    estimator_features = getattr(estimator, "feature_names_in_", None)
    if estimator_features is None:
        raise ValueError(f"Model artifact {path} has no named feature schema; retrain it with scripts/05_train_model.py.")
    actual = list(estimator_features)
    if actual != feature_names:
        raise ValueError(
            f"Model artifact {path} feature schema mismatch: expected {feature_names}, got {actual}."
        )
    feature_count = getattr(estimator, "n_features_in_", len(actual))
    if feature_count != len(feature_names):
        raise ValueError(
            f"Model artifact {path} expects {feature_count} features; contract requires {len(feature_names)}."
        )


def load_model_artifact(path: Path) -> dict:
    """Load a versioned artifact or a legacy estimator with an exact named schema."""
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        if artifact.get("format") != MODEL_ARTIFACT_FORMAT:
            raise ValueError(f"Unsupported model artifact format in {path}.")
        if artifact.get("version") != MODEL_ARTIFACT_VERSION:
            raise ValueError(f"Unsupported model artifact version in {path}: {artifact.get('version')!r}.")
        if artifact.get("feature_contract_version") != FEATURE_CONTRACT_VERSION:
            raise ValueError(
                f"Model artifact {path} uses an incompatible feature contract version: "
                f"{artifact.get('feature_contract_version')!r}."
            )
        feature_names = artifact.get("feature_names")
        if feature_names != FEATURE_COLUMNS:
            raise ValueError(
                f"Model artifact {path} feature schema mismatch: expected {FEATURE_COLUMNS}, got {feature_names}."
            )
        if artifact.get("label_mapping") != LABEL_MAPPING:
            raise ValueError(f"Model artifact {path} label mapping does not match the current contract.")
        estimator = artifact.get("estimator")
        if estimator is None or not callable(getattr(estimator, "predict", None)):
            raise ValueError(f"Model artifact {path} does not contain a predictive estimator.")
        _validate_estimator_schema(estimator, feature_names, path)
        return artifact

    _validate_estimator_schema(artifact, FEATURE_COLUMNS, path)
    return make_model_artifact(artifact, horizon=0)


def save_model_artifact(estimator, path: Path, horizon: int) -> None:
    """Persist a versioned estimator artifact at the training/deployment path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(make_model_artifact(estimator, horizon), path)
