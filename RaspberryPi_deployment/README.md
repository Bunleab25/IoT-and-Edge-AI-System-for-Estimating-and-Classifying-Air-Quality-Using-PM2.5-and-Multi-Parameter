# Raspberry Pi Deployment

Flask inference service for AeroSense RMUTI. For the project overview, data and results, see the [root README](../README.md).

## Contents

| File | Purpose |
|---|---|
| [`model.py`](model.py) | Flask API (`/predict`, `/health`) on port 5001 |
| `rf_day1.pkl`, `rf_day2.pkl`, `rf_day3.pkl` | RandomForest models, one per horizon (45 features) |
| `aqi_nowcast_model_smote.pkl` | Nowcast model trained with SMOTE |
| [`test.py`](test.py) | Prints the feature names a model expects |

## Run

```bash
pip install -r ../requirements.txt
python model.py
```

The service keeps a rolling history in `live_sensor_log.csv`, created on first request.

## API

`POST /predict`

```json
{ "pm25": 42.3, "temperature": 28.5, "humidity": 65, "pressure": 995 }
```

Returns `predicted_category`, `confidence`, `class_probabilities` and `feature_confidence` for `day1`, `day2` and `day3`.

`GET /health` returns model-loading status and the history row count.

## Behavior notes

- Day+1 uses real history. Day+2 and Day+3 use forecast weather from Open-Meteo, with PM2.5 persisted from today.
- If the weather API fails, today's weather is reused and `feature_confidence` is reported as low.
