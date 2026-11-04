# AeroSense: Edge AI Air-Quality Forecasting

AeroSense is an air-quality forecasting project for the RMUTI campus in Nakhon Ratchasima, Thailand. Its Python pipeline combines daily PM2.5 observations with weather measurements, engineers chronological features, and trains three AQI-category classifiers for one-, two-, and three-day horizons. A Flask API is provided for Raspberry Pi deployment, with an Arduino sensor sketch and a sanitized Node-RED flow included in the repository.

The project is a research and deployment prototype, not a fully packaged production service. Model binaries are generated locally by training and are intentionally excluded from version control; a fresh checkout must train compatible artifacts before edge inference.

![AeroSense edge AI forecast preview](ready.jpeg)

## System architecture

```text
OpenAQ daily PM2.5 + Open-Meteo weather
                    |
                    v
   Raw daily dataset (data/raw/)
                    |
          clean and validate
                    v
     Chronological observations
                    |
       shared causal feature builder
                    v
       38 features + day+1..3 labels
                    |
     chronological split + 3-row purge
                    |
  purged TimeSeriesSplit model search
                    |
      versioned best_day1..3 artifacts
             /                 \
            v                   v
 Evaluation reports       Forecast scripts
Sensors --> Raspberry Pi Flask API --> three-day AQI forecast
                |       ^
                |       +-- Open-Meteo forecast weather
                +---------- local daily observation history
```

The data collection script uses OpenAQ daily PM2.5 data and Open-Meteo archive weather. Cleaning validates required columns, orders dates, forward-fills at most three consecutive missing measurements from the past, and derives AQI categories using the project's four-class thresholds: Very Good (up to 15 µg/m³), Good (up to 25), Moderate (up to 37.5), and Unhealthy (above 37.5). The feature and label pipeline preserves date order and attaches future targets by exact calendar date, not merely the next row; dates without an observation at each target horizon are excluded. The checked-in CSVs are small (the raw file is about 116 KB); no additional sample data is copied by the demo. Check the upstream OpenAQ and Open-Meteo data terms and attribution requirements before redistributing source observations. The repository MIT license does not supersede third-party data terms.

## Repository structure

The repository separates installable Python components from runnable workflow scripts and keeps experiment inputs, datasets, reports, and optional deployment assets in distinct areas:

```text
.
|-- src/aerosense/                    Reusable Python package
|   |-- model_contract.py             Ordered features, labels, artifact validation
|   |-- evaluation_metrics.py         Shared classification metrics
|   `-- experiment_utils.py           Config loading and run provenance
|-- scripts/                          Numbered pipeline commands and offline demo
|   |-- 00_collect_data.py ... 07_forecast_pipeline.py
|   |-- run_demo.py
|   `-- _bootstrap.py                 Makes src package work from direct script runs
|-- configs/experiment.json           Seeds, splits, CV and search parameters
|-- data/                             Raw, cleaned and chronological model-ready CSVs
|-- outputs/                          Reports, provenance, feature details and plots
|-- tests/                            Contract, data-client and demo tests
|-- RaspberryPi_deployment/           Optional Flask edge inference and docs
|-- Arduino/                          Optional sensor sketch
|-- node-red/                         Sanitized optional flow export
|-- models/                           Locally generated, Git-ignored model artifacts
|-- pyproject.toml                    Src-layout package and pytest configuration
|-- requirements.txt                  Pinned runtime and test dependencies
|-- README.md
|-- LICENSE
`-- ready.jpeg                        Project preview image
```

`scripts/00_collect_data.py` through `scripts/07_forecast_pipeline.py` remain the stable command-line workflow; reusable logic lives in `src/aerosense/`. `scripts/_bootstrap.py` preserves direct `python scripts/...` execution without requiring an editable install. Generated model files and transient Python caches are not part of the tracked source layout.

## Environment setup

Create an isolated environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

The training configuration lives in `configs/experiment.json`. Direct dependencies are pinned in `requirements.txt`; run provenance records resolved package versions. Training records config and input hashes, git revision/dirty state, source hashes, split dates and sizes, selected models, and search parameters in `outputs/training_run.json`. The configured search evaluates 12 parameter samples per model and horizon over five expanding chronological folds (six searches total); search-level parallelism is capped at four workers and each estimator uses one thread. Use the same config, data hashes, and dependency versions when comparing runs; the manifest improves reproducibility but does not guarantee bit-for-bit identity across CPUs or native library builds.

## Run the offline demo

The demo needs no API keys, network access, model artifacts, or extra sample data. It rebuilds features from the checked-in cleaned data, trains a small seeded Random Forest per horizon on the earlier chronological partition, applies the three-row target embargo, and compares against persistence. It writes detailed metrics and provenance to `outputs/demo_report.json`:

```bash
python scripts/run_demo.py
```

This is a runnable software demonstration, not the tuned model candidate and not a substitute for the configured XGBoost/LightGBM experiment. In the verified run (seed 42, 869 training rows, 218 test rows), macro-F1 was 0.626 / 0.572 / 0.547 for the demo Random Forest at day +1 / +2 / +3, compared with persistence at 0.663 / 0.518 / 0.494. These are demo-only holdout results, not the tuned model's evaluation.

## Run the data and model pipeline

The repository includes raw and cleaned datasets, so data collection can be skipped when using the checked-in data. To fetch fresh data, set an OpenAQ API key and run the collector; it also requests archive weather from Open-Meteo:

```bash
export OPENAQ_API_KEY="your-openaq-api-key"
python scripts/00_collect_data.py
```

Run the remaining stages in order. The collector can be skipped when using the checked-in raw data; begin at cleaning. To create a fresh dataset, set `OPENAQ_API_KEY` and run the collector first.

```bash
python scripts/01_data_cleaning.py
python scripts/02_eda.py
python scripts/03_feature_engineering.py
python scripts/04_train_test_split.py
python scripts/05_train_model.py
python scripts/06_evaluate.py
python scripts/07_forecast_pipeline.py
```

Feature generation writes the engineered rows to `data/model_ready/train_test_shifted.csv` and records the ordered feature contract in `outputs/feature_contract.json`. Future targets are matched by exact calendar date; rows whose target date is absent are excluded, so a dataset gap cannot silently turn a multi-day outcome into a nominal next-day label. The split starts from the configured 80/20 chronological boundary and removes the three rows immediately before the holdout; their day+1 through day+3 targets could otherwise include holdout-period outcomes. Hyperparameter selection uses expanding-window `TimeSeriesSplit` with the same three-row gap. XGBoost and LightGBM are tuned per horizon with the fixed seed and macro-F1 selection criterion in `configs/experiment.json`; the best selected estimators and search details are saved as:

```text
models/best_day1.pkl
models/best_day2.pkl
models/best_day3.pkl
```

Each selected-model artifact includes the estimator, its feature names, label mapping, and contract version. The best XGBoost and LightGBM candidates are also saved as versioned comparison artifacts under `models/xgb_day*.pkl` and `models/lgbm_day*.pkl`. Evaluation and forecast inference load and validate these artifacts rather than assuming a feature list.

Evaluation uses only the later, purged chronological holdout for final comparison. It reports accuracy, balanced accuracy, macro/weighted F1, Matthews correlation, log loss, multiclass Brier score, per-class precision/recall/F1/support, and confusion matrices for XGBoost, LightGBM, the seeded demo Random Forest, persistence (today's category), and the most frequent training label. The evaluator verifies the demo report was produced from the same data, config, and holdout date ranges before including it. `outputs/evaluation_report.json` records results, search parameters, train/test ranges and hashes, settings, environment and source fingerprints. `outputs/model_comparison.json` contains CV search scores and selected parameters. The evaluator fails if the split overlaps or lacks the required embargo.

### Boosting-model comparison (current reproducible run)

The configured search completed all six randomized searches (12 candidates × 5 purged chronological folds for each model/horizon) in about 63 seconds in the latest run. All results below use the same 869-row training partition and 218-row holdout (2025-11-30 through 2026-08-04); the 3-row embargo is applied before the holdout. Values are macro-F1; full metrics and provenance are in `outputs/evaluation_report.json`, and CV selection results are in `outputs/model_comparison.json`.

| Horizon | XGBoost CV | LightGBM CV | XGBoost holdout | LightGBM holdout | Demo RF holdout | Persistence | Training-majority |
|---|---:|---:|---:|---:|---:|---:|---:|
| Day +1 | 0.607 | 0.577 | 0.698 | 0.694 | 0.626 | 0.663 | 0.111 |
| Day +2 | 0.481 | 0.473 | 0.562 | 0.563 | 0.572 | 0.518 | 0.109 |
| Day +3 | 0.426 | 0.395 | 0.533 | 0.492 | 0.547 | 0.494 | 0.109 |

These are a single chronological holdout comparison, not a significance test or production-performance claim. XGBoost led CV at each horizon; the holdout is mixed: persistence is strongest at day +1, the demo forest slightly leads at day +2 and day +3, while XGBoost and LightGBM remain competitive. The report includes accuracy, balanced accuracy, weighted F1, Matthews correlation, log loss, multiclass Brier score, per-class metrics, and confusion matrices for each comparison model.

## Run the edge API

Once training has produced all three `models/best_day*.pkl` artifacts, start the service from the repository root:

```bash
python RaspberryPi_deployment/model.py
```

The API provides `GET /health` and `POST /predict`. Health reports which horizon models loaded. Prediction expects numeric `pm25`, `temperature`, `humidity`, and `pressure` values, plus `windspeed` and `winddirection` unless those readings are already available in the local history log. The service needs at least three prior daily observations for lag features; on a cold start, collect that history before expecting a forecast. Older history logs that lack wind columns must be rebuilt.

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

The response retains a `forecast` object with `day1`, `day2`, and `day3` entries, plus `forecast_fetch_error` and `input_received`. Each forecast includes a predicted category, confidence, class probabilities, and feature-confidence note. For day +2 and day +3 the service uses Open-Meteo weather forecasts and a persistence estimate for future PM2.5; unavailable weather is reported and the latest sensor readings are reused.

## Pipeline alignment and current status

The aligned implementation addresses the previous schema split between training, evaluation, and edge inference:

- `src/aerosense/model_contract.py` defines the ordered 38-feature schema, causal feature construction, AQI label mapping, and versioned artifact validation.
- `scripts/03_feature_engineering.py` uses that shared builder and writes feature-contract metadata. Stagnation normalization is calculated against preceding observations so later data does not influence earlier rows.
- Training validates the prepared dataset contract, and evaluation and batch forecasting use the same schema and verify model artifacts.
- The Raspberry Pi service loads the aligned `models/best_day*.pkl` artifacts and rejects incompatible schemas. It does not silently substitute the older `RaspberryPi_deployment/rf_day*.pkl` files.
- The existing `/health` and `/predict` routes and forecast response fields are retained, with explicit model, input, and history validation.

Cleaning now imputes short gaps with past-only forward fill instead of two-sided interpolation, avoiding future observations influencing earlier features. Forecast targets use exact +1/+2/+3 calendar-day observations, while the chronological holdout and each CV validation boundary retain a conservative three-row embargo to prevent future labels from crossing into later evaluation periods.

No model binaries are checked into version control. The current local training run generated the aligned `models/best_day*.pkl`, `models/xgb_day*.pkl`, and `models/lgbm_day*.pkl` files (ignored by Git); these allow the edge API to run in this checkout, while a clean clone must run training before edge inference. Legacy pickle files under `RaspberryPi_deployment/` do not satisfy the current artifact contract and are intentionally not loaded. The demo remains runnable without trained model artifacts. The checked-in evaluation report and model comparison JSON correspond to the current run; older plots and snapshots are retained as historical outputs.

To run the tests:

```bash
python -m pytest -q
```

The suite covers causal feature construction and imputation, split embargo, artifact round-tripping and schema rejection, the Flask API response/health routes, and the offline demo. The full hyperparameter search is separate from the test suite; it was run for the comparison above and is not run by the offline demo.

## Optional hardware integration

`Arduino/air_quality_sensor.ino` and `node-red/aerosense_edge_flow_sanitized.json` are optional integration materials. The validated software workflow is the Python data/model pipeline, offline demo, and Flask API. Hardware-specific wiring, broker settings, and end-to-end MQTT behavior are not validated by the Python test suite; configure and verify them for the target device before deployment.

## License

See [LICENSE](LICENSE).
