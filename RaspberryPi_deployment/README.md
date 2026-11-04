# Edge AI Air-Quality Forecast System

AeroSense RMUTI is a campus-focused air-quality intelligence system designed for deployment on a Raspberry Pi 5 at the edge. It combines local sensor measurements, short-term weather context, and trained AQI forecasting models to estimate air-quality conditions for the next one to three days for the RMUTI campus in Nakhon Ratchasima, Thailand.

This is no longer just a research prototype. It is structured as an edge AI system with:

- local real-time inference at the device edge
- sensor-based input collection
- AQI forecasting for short horizons
- lightweight model serving on Raspberry Pi 5
- future-ready integration with ESP32-based environmental monitoring

## Scenario

The system models an air-quality monitoring scenario for RMUTI campus in Nakhon Ratchasima, Thailand. It is intended for edge deployment where a small computing device runs inference close to the source of sensor data, reduces cloud dependency, and provides timely AQI information to local users or campus services.

## Edge AI architecture

```text
ESP32 / sensor node
        │
        ▼
Raspberry Pi 5 (edge device)
        │
        ├── local sensor ingestion
        ├── AQI feature preprocessing
        ├── ML inference model
        ├── short-term forecast logic
        └── local API / dashboard output
```

The edge service in this folder is implemented in [model.py](model.py). It accepts sensor readings, builds the same ordered 38-feature set used for training, runs the trained models, and returns AQI predictions for the next three days.

## Model artifacts and input history

Run the repository training pipeline to create `models/best_day1.pkl` through `models/best_day3.pkl`. Each artifact includes its feature schema and label mapping; this service validates both at startup and does not fall back to older `rf_day*.pkl` files. Retrain models when the feature contract changes.

The API needs three prior daily observations to construct lag features. A cold start must supply PM2.5, temperature, humidity, pressure, wind speed, and wind direction, and collect at least three prior daily readings before forecasts can be produced. Wind speed and direction may be omitted from later requests when available in the local history log. Older history logs without wind measurements must be rebuilt with the new columns before starting the service.

## Setup and service behavior

From the repository root, create the environment, install the pinned dependencies, regenerate compatible data splits, train the horizon models, then start the service:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/03_feature_engineering.py
python scripts/04_train_test_split.py
python scripts/05_train_model.py
python RaspberryPi_deployment/model.py
```

Training is a separate, comparatively expensive step. The service expects the resulting files in the repository-level `models/` directory, while the observation history is stored in this deployment folder as `live_sensor_log.csv`. Model binaries are generated locally and ignored by Git; `/health` reports models unavailable in a fresh checkout until training has produced compatible artifacts.

`GET /health` reports model load status and history length. `POST /predict` accepts the fields shown below and returns predictions for day +1 through day +3. The service writes each submitted reading to the local history log, obtains weather context for later forecast horizons from Open-Meteo, and falls back to the latest sensor readings if the forecast request is unavailable. This endpoint is an inference interface; it does not directly read the Arduino sensor or configure MQTT/Node-RED.

For local verification from the repository root, run `python -m pytest -q`. The Python tests cover the API using mock estimators and weather responses; they do not verify Raspberry Pi hardware, sensor wiring, network access, or MQTT integration.

## What this repo contains

- [scripts/00_collect_data.py](../scripts/00_collect_data.py): data collection from OpenAQ and weather sources
- [scripts/01_data_cleaning.py](../scripts/01_data_cleaning.py): data cleaning and labeling
- [scripts/03_feature_engineering.py](../scripts/03_feature_engineering.py): lag, rolling, and time features
- [scripts/05_train_model.py](../scripts/05_train_model.py): training and model comparison
- [scripts/06_evaluate.py](../scripts/06_evaluate.py): evaluation metrics and reporting
- [scripts/07_forecast_pipeline.py](../scripts/07_forecast_pipeline.py): forecast generation pipeline
- [src/aerosense/model_contract.py](../src/aerosense/model_contract.py): shared features, labels, and artifact validation
- [src/aerosense/experiment_utils.py](../src/aerosense/experiment_utils.py): experiment configuration and provenance
- [model.py](model.py): Flask-based edge inference API for Raspberry Pi 5
- [tests/test_openaq_parser.py](../tests/test_openaq_parser.py): validation for the data pipeline

## AI and deployment goal

This project is designed to support the following real-world edge scenario:

1. Read local environmental conditions from a sensor or input source
2. Build the required AQI feature context on-device
3. Run inference with a trained model on Raspberry Pi 5
4. Forecast air-quality category for Day +1, Day +2, and Day +3
5. Expose the result through an API or dashboard endpoint

## Quick start

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the training pipeline

```bash
python scripts/00_collect_data.py
python scripts/01_data_cleaning.py
python scripts/02_eda.py
python scripts/03_feature_engineering.py
python scripts/04_train_test_split.py
python scripts/05_train_model.py
python scripts/06_evaluate.py
python scripts/07_forecast_pipeline.py
```

### Run the edge inference service

```bash
python RaspberryPi_deployment/model.py
```

After compatible models are trained and at least three prior daily readings have been accumulated, send a request to the local API. A cold start should include wind speed and direction:

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pm25": 42.3,
    "temperature": 28.5,
    "humidity": 65,
    "pressure": 995,
    "windspeed": 3.2,
    "winddirection": 90
  }'
```

## Hardware target

This repository is intended for a Raspberry Pi 5 edge deployment, with the possibility of pairing with an ESP32 sensor node. The current code already follows the structure of a local inference service intended to run close to the monitored environment.

## Project status

- Research and data pipeline: complete
- Training and evaluation pipeline: complete
- Edge inference prototype: complete
- Production-style deployment packaging: next step

## Notes

- The project intentionally avoids random train/test shuffling and uses chronological evaluation.
- AQI categories are mapped into a compact four-class scheme for improved stability.
- Forecast fallback logic is included so the system can continue operating even when external weather APIs are temporarily unavailable.

## License

This project is distributed under the MIT License.
