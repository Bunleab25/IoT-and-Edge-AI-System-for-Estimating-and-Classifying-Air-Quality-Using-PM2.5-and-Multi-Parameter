# Raspberry Pi Deployment

Flask inference service for AeroSense RMUTI. For the project overview, data and results, see the [root README](../README.md).

## Files

| File | Purpose |
|---|---|
| [`model.py`](model.py) | Flask API (`/predict`, `/health`) on port 5001 |
| [`test.py`](test.py) | Helper script for inspecting a model |

## Prerequisites

Train the models from the repository root. The service reads `models/best_day1.pkl` to `models/best_day3.pkl`, which are Git-ignored. Each artifact carries its feature schema and label mapping, and the service validates both at startup. It does not fall back to the older `rf_day*.pkl` files, and models must be retrained whenever the feature contract changes.

```bash
python scripts/05_train_model.py
```

## Run

```bash
python RaspberryPi_deployment/model.py
```

The service keeps a rolling daily history in a local log and needs at least 3 prior daily observations for lag features. On a cold start, send wind speed and direction with the first requests and collect enough history before expecting a forecast. Older history logs without wind columns must be rebuilt.

## API

`POST /predict`

```json
{ "pm25": 42.3, "temperature": 28.5, "humidity": 65, "pressure": 995,
  "windspeed": 3.2, "winddirection": 90 }
```

`windspeed` and `winddirection` can be omitted only if they are already in the local history. The response has a `forecast` object with `day1`, `day2` and `day3`, plus `forecast_fetch_error` and `input_received`. Each forecast includes a category, confidence, class probabilities and a feature-confidence note.

`GET /health` reports which horizon models loaded.

## Behavior notes

- Day+2 and Day+3 use Open-Meteo forecast weather, with PM2.5 persisted from today.
- If the weather API fails, the latest readings are reused and the failure is reported.
