"""Flask inference API using the shared training feature and artifact contract."""
from datetime import datetime, timedelta
from pathlib import Path
import sys

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from aerosense.model_contract import (  # noqa: E402
    OBSERVATION_COLUMNS,
    TARGET_LABELS,
    engineer_features,
    load_model_artifact,
    prepare_feature_frame,
)

app = Flask(__name__)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATHS = {f"day{horizon}": MODEL_DIR / f"best_day{horizon}.pkl" for horizon in range(1, 4)}
HISTORY_LOG_PATH = Path(__file__).resolve().parent / "live_sensor_log.csv"
MAX_HISTORY = 30
RMUTI_LAT, RMUTI_LON = 14.987471, 102.117965

models = {}
model_errors = {}
for horizon, path in MODEL_PATHS.items():
    try:
        models[horizon] = load_model_artifact(path)["estimator"]
        print(f"Loaded compatible {horizon} model from {path}")
    except FileNotFoundError as exc:
        models[horizon] = None
        model_errors[horizon] = str(exc)
        print(f"ERROR: {exc}")
    except ValueError as exc:
        models[horizon] = None
        model_errors[horizon] = str(exc)
        print(f"ERROR: {exc}")


def load_history() -> pd.DataFrame:
    """Load and validate persisted sensor observations."""
    columns = ["date", *OBSERVATION_COLUMNS]
    if not HISTORY_LOG_PATH.exists():
        print(f"WARNING: {HISTORY_LOG_PATH} not found. Starting with an empty buffer.")
        return pd.DataFrame(columns=columns)

    history = pd.read_csv(HISTORY_LOG_PATH, parse_dates=["date"])
    missing = sorted(set(columns).difference(history.columns))
    if missing:
        raise ValueError(f"History log is missing required columns {missing}; provide a log with wind measurements.")
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    if history["date"].isna().any():
        raise ValueError(f"History log {HISTORY_LOG_PATH} contains invalid dates.")
    for column in OBSERVATION_COLUMNS:
        history[column] = pd.to_numeric(history[column], errors="raise")
    if history[OBSERVATION_COLUMNS].isna().any().any():
        raise ValueError(f"History log {HISTORY_LOG_PATH} contains missing sensor readings.")
    return history.sort_values("date").tail(MAX_HISTORY).reset_index(drop=True)


history_df = load_history()


def append_and_save(row: dict) -> None:
    """Add or update today's observation and persist the bounded history."""
    global history_df
    row_date = pd.to_datetime(row["date"])
    row["date"] = row_date
    history_df["date"] = pd.to_datetime(history_df["date"])
    matching = history_df.index[history_df["date"].eq(row_date)]
    if len(matching):
        for key, value in row.items():
            history_df.at[matching[0], key] = value
    else:
        history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
    history_df = history_df.sort_values("date").tail(MAX_HISTORY).reset_index(drop=True)
    history_df.to_csv(HISTORY_LOG_PATH, index=False)


def circular_mean_degrees(values: pd.Series) -> float:
    """Return a circular mean of wind directions in degrees."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    radians = np.deg2rad(clean.to_numpy())
    angle = np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    return float((angle + 360.0) % 360.0)


def fetch_forecast_weather() -> dict:
    """Fetch daily weather context, including wind, for forecast horizons."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": RMUTI_LAT,
            "longitude": RMUTI_LON,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
            "forecast_days": 4,
            "timezone": "Asia/Bangkok",
        },
        timeout=15,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    weather = pd.DataFrame(hourly)
    weather["date"] = pd.to_datetime(weather["time"]).dt.strftime("%Y-%m-%d")
    weather = weather.rename(
        columns={"wind_speed_10m": "windspeed_10m", "wind_direction_10m": "winddirection_10m"}
    )
    daily = {}
    for date, group in weather.groupby("date"):
        daily[date] = {
            "temperature_2m": float(group["temperature_2m"].mean()),
            "relative_humidity_2m": float(group["relative_humidity_2m"].mean()),
            "surface_pressure": float(group["surface_pressure"].mean()),
            "windspeed_10m": float(group["windspeed_10m"].mean()),
            "winddirection_10m": circular_mean_degrees(group["winddirection_10m"]),
        }
    return daily


def build_feature_row(observations: pd.DataFrame) -> pd.DataFrame:
    """Build the latest row using the exact same feature engineering as training."""
    engineered = engineer_features(observations)
    return prepare_feature_frame(engineered.iloc[[-1]])


@app.route("/health", methods=["GET"])
def health():
    loaded = {key: value is not None for key, value in models.items()}
    return jsonify({
        "status": "ok" if all(loaded.values()) else "error",
        "models_loaded": loaded,
        "history_rows": len(history_df),
        "model_errors": model_errors,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Return AQI categories and class probabilities for the next three days."""
    if any(model is None for model in models.values()):
        return jsonify({
            "error": "One or more models failed to load or are incompatible.",
            "model_errors": model_errors,
        }), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    required_inputs = {
        "pm25": "pm25",
        "temperature": "temperature_2m",
        "humidity": "relative_humidity_2m",
        "pressure": "surface_pressure",
    }
    try:
        values = {column: float(data[key]) for key, column in required_inputs.items()}
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid or missing input: {exc}"}), 400
    if not np.isfinite(list(values.values())).all():
        return jsonify({"error": "Sensor readings must be finite numbers."}), 400
    if values["pm25"] < 0 or values["relative_humidity_2m"] < 0 or values["relative_humidity_2m"] > 100:
        return jsonify({"error": "PM2.5 must be non-negative and humidity must be between 0 and 100."}), 400
    if values["surface_pressure"] <= 0:
        return jsonify({"error": "Pressure must be a positive number."}), 400

    for key, column in (("windspeed", "windspeed_10m"), ("winddirection", "winddirection_10m")):
        try:
            values[column] = float(data[key]) if key in data else float(history_df[column].iloc[-1])
        except (ValueError, IndexError):
            return jsonify({
                "error": f"Missing {key}; supply both windspeed and winddirection for a cold start."
            }), 400
    if not np.isfinite([values["windspeed_10m"], values["winddirection_10m"]]).all():
        return jsonify({"error": "Wind readings must be finite numbers."}), 400
    if values["windspeed_10m"] < 0 or not 0 <= values["winddirection_10m"] <= 360:
        return jsonify({"error": "Wind speed must be non-negative and direction must be between 0 and 360."}), 400

    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    today_observation = {"date": today_str, **values}
    try:
        append_and_save(today_observation)
    except OSError as exc:
        return jsonify({"error": f"Could not persist sensor history: {exc}"}), 500

    if len(history_df) < 4:
        return jsonify({
            "error": "At least three prior daily sensor observations are required to build lag features."
        }), 400

    rows = {"day1": build_feature_row(history_df)}
    confidence_notes = {"day1": "high"}
    forecast_error = None
    try:
        weather_by_date = fetch_forecast_weather()
    except requests.exceptions.RequestException as exc:
        forecast_error = str(exc)
        weather_by_date = {}

    for horizon in (2, 3):
        forecast_date = (today + timedelta(days=horizon)).strftime("%Y-%m-%d")
        forecast = weather_by_date.get(forecast_date, {})
        if not forecast:
            missing_forecast = f"No weather forecast available for {forecast_date}; using latest sensor readings."
            forecast_error = "; ".join(filter(None, (forecast_error, missing_forecast)))
            confidence_notes[f"day{horizon}"] = "low (weather unavailable; latest sensor readings reused)"
        else:
            confidence_notes[f"day{horizon}"] = "mixed (weather forecasted; PM2.5 persisted)"
        pseudo_row = {
            "date": forecast_date,
            "pm25": values["pm25"],
            "temperature_2m": forecast.get("temperature_2m", values["temperature_2m"]),
            "relative_humidity_2m": forecast.get("relative_humidity_2m", values["relative_humidity_2m"]),
            "surface_pressure": forecast.get("surface_pressure", values["surface_pressure"]),
            "windspeed_10m": forecast.get("windspeed_10m", values["windspeed_10m"]),
            "winddirection_10m": forecast.get("winddirection_10m", values["winddirection_10m"]),
        }
        source = pd.concat([history_df, pd.DataFrame([pseudo_row])], ignore_index=True)
        rows[f"day{horizon}"] = build_feature_row(source)

    results = {}
    for horizon, features in rows.items():
        estimator = models[horizon]
        try:
            prediction = int(estimator.predict(features)[0])
            probabilities = estimator.predict_proba(features)[0]
            class_probabilities = {
                TARGET_LABELS[int(label)]: round(float(probability), 3)
                for label, probability in zip(estimator.classes_, probabilities)
            }
        except (ValueError, IndexError, KeyError) as exc:
            return jsonify({"error": f"Inference failed for {horizon}: {exc}"}), 500
        results[horizon] = {
            "predicted_category": TARGET_LABELS[prediction],
            "confidence": round(float(max(probabilities)), 3),
            "class_probabilities": class_probabilities,
            "feature_confidence": confidence_notes[horizon],
        }

    return jsonify({
        "forecast": results,
        "forecast_fetch_error": forecast_error,
        "input_received": {
            "pm25": values["pm25"],
            "temperature": values["temperature_2m"],
            "humidity": values["relative_humidity_2m"],
            "pressure": values["surface_pressure"],
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
