"""Fetch Open-Meteo weather forecasts and create the next three AQI predictions."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import _bootstrap  # noqa: F401
from aerosense.model_contract import (
    LABEL_MAPPING,
    engineer_features,
    load_model_artifact,
    pm25_to_category,
    prepare_feature_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_CSV = PROJECT_ROOT / "data/processed/cleaned_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_JSON = OUTPUTS_DIR / "forecast_latest.json"
LATITUDE, LONGITUDE = 14.987471, 102.117965
TIMEZONE = "Asia/Bangkok"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def circular_mean_degrees(values: pd.Series) -> float:
    """Return the circular mean of wind direction readings in degrees (0-360)."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    radians = np.deg2rad(clean.to_numpy())
    mean_angle = np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float((mean_angle + 360.0) % 360.0)


def load_history() -> pd.DataFrame:
    """Load recent chronological rows from cleaned data for lag and rolling calculations."""
    if not CLEANED_CSV.exists():
        raise FileNotFoundError("Run 01_data_cleaning.py first; it supplies the latest historical PM2.5.")
    history = pd.read_csv(CLEANED_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if len(history) < 7:
        raise ValueError("Need at least 7 historical rows to build lag/rolling PM2.5 features.")
    return history.tail(30).reset_index(drop=True)


def fetch_daily_weather() -> pd.DataFrame:
    """Fetch hourly forecast data and average it into the next three local days."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
        "forecast_days": 4,
        "timezone": TIMEZONE,
    }
    print("Requesting forecast weather from Open-Meteo FORECAST API...")
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=30)
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        weather = pd.DataFrame(hourly)
        if weather.empty:
            raise ValueError("Open-Meteo returned no hourly forecast values.")
        weather = weather.rename(columns={"wind_speed_10m": "windspeed_10m", "wind_direction_10m": "winddirection_10m"})
        weather["time"] = pd.to_datetime(weather["time"])
        weather["date"] = weather["time"].dt.date
        daily = weather.groupby("date", as_index=False).agg(
            temperature_2m=("temperature_2m", "mean"),
            relative_humidity_2m=("relative_humidity_2m", "mean"),
            surface_pressure=("surface_pressure", "mean"),
            windspeed_10m=("windspeed_10m", "mean"),
            winddirection_10m=("winddirection_10m", circular_mean_degrees),
        )
        tomorrow = datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=1)
        wanted_dates = [tomorrow + timedelta(days=offset) for offset in range(3)]
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily[daily["date"].dt.date.isin(wanted_dates)].sort_values("date").reset_index(drop=True)
        if len(daily) == 3:
            return daily
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"Forecast API unavailable or incomplete ({exc}); using recent historical weather as fallback.")

    history = pd.read_csv(CLEANED_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    latest = history.iloc[-1].copy()
    tomorrow = datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=1)
    fallback_rows = []
    for offset in range(3):
        target_date = tomorrow + timedelta(days=offset)
        fallback_rows.append({
            "date": pd.Timestamp(target_date),
            "temperature_2m": float(latest["temperature_2m"]),
            "relative_humidity_2m": float(latest["relative_humidity_2m"]),
            "surface_pressure": float(latest["surface_pressure"]),
            "windspeed_10m": float(latest["windspeed_10m"]),
            "winddirection_10m": float(latest["winddirection_10m"]),
        })
    return pd.DataFrame(fallback_rows).sort_values("date").reset_index(drop=True)


def build_feature_row(history: pd.DataFrame, weather: pd.DataFrame, horizon: int, persisted_pm25: float) -> pd.Series:
    """Build a forecast row with the same causal feature builder used for training."""
    future = weather.iloc[:horizon][
        ["date", "temperature_2m", "relative_humidity_2m", "surface_pressure", "windspeed_10m", "winddirection_10m"]
    ].copy()
    future["pm25"] = persisted_pm25
    observations = pd.concat([history, future], ignore_index=True)
    engineered = engineer_features(observations)
    return prepare_feature_frame(engineered.iloc[[-1]]).iloc[0]


def main() -> None:
    """Combine forecast weather with a persistence-based PM2.5 approximation and write AQI forecasts."""
    if not CLEANED_CSV.exists():
        raise FileNotFoundError("Run 01_data_cleaning.py first; it supplies the latest observed PM2.5.")

    history = load_history()
    latest_pm25 = float(history["pm25"].iloc[-1])
    latest_category = pm25_to_category(latest_pm25)
    reverse_map = {value: label for label, value in LABEL_MAPPING.items()}
    print(f"Using latest observed PM2.5 ({latest_pm25:.2f} µg/m³) as the persistence anchor.")
    print(f"Latest observed AQI category: {latest_category}")

    weather = fetch_daily_weather()
    output = {}
    for horizon in range(1, 4):
        model_path = MODELS_DIR / f"best_day{horizon}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}; run 05_train_model.py first.")

        forecast_day = weather.iloc[horizon - 1]
        feature_row = build_feature_row(history, weather, horizon, latest_pm25)

        artifact = load_model_artifact(model_path)
        model = artifact["estimator"]
        prediction = int(model.predict(feature_row.to_frame().T)[0])

        confidence = "high (wind/temp/humidity forecasted; PM2.5 persistence remains low)" if horizon > 1 else "high"
        output[f"day{horizon}"] = {
            "date": str(forecast_day["date"].date()),
            "predicted_aqi_category": reverse_map.get(prediction, str(prediction)),
            "feature_confidence": confidence,
        }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved latest forecast: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
