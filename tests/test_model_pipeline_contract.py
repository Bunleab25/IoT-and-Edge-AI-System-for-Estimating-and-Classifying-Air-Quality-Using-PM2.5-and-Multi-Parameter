import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from aerosense import model_contract
from aerosense.experiment_utils import load_experiment_config


def load_edge_module():
    path = ROOT / "RaspberryPi_deployment" / "model.py"
    spec = importlib.util.spec_from_file_location("edge_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def observations(rows=40):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    index = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "pm25": 10 + index * 0.8,
        "temperature_2m": 22 + index * 0.1,
        "relative_humidity_2m": 70 - index * 0.2,
        "surface_pressure": 1000 + np.sin(index / 3),
        "windspeed_10m": 4 + np.cos(index / 4),
        "winddirection_10m": (index * 13) % 360,
    })


class FakeEstimator:
    def __init__(self):
        self.feature_names_in_ = np.asarray(model_contract.FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(model_contract.FEATURE_COLUMNS)
        self.classes_ = np.asarray([0, 1, 2, 3])
        self.last_features = None

    def predict(self, features):
        self.last_features = features.copy()
        return np.asarray([2])

    def predict_proba(self, features):
        self.last_features = features.copy()
        return np.asarray([[0.1, 0.2, 0.6, 0.1]])


class ModelPipelineContractTests(unittest.TestCase):
    def test_feature_builder_is_ordered_and_causal(self):
        raw = observations()
        engineered = model_contract.engineer_features(raw)
        prepared = model_contract.prepare_feature_frame(engineered.iloc[3:])
        self.assertEqual(prepared.columns.tolist(), model_contract.FEATURE_COLUMNS)
        self.assertEqual(prepared.shape, (len(raw) - 3, 38))

        changed = raw.copy()
        changed.loc[39, "surface_pressure"] += 500
        changed_features = model_contract.engineer_features(changed)
        self.assertAlmostEqual(
            engineered.loc[38, "stagnation_score"],
            changed_features.loc[38, "stagnation_score"],
        )
        pd.testing.assert_frame_equal(
            engineered.loc[:38, model_contract.FEATURE_COLUMNS],
            changed_features.loc[:38, model_contract.FEATURE_COLUMNS],
        )

    def test_future_labels_match_calendar_dates_across_gaps(self):
        data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]),
            "pm25": [10.0, 20.0, 30.0, 40.0],
        })
        labeled = model_contract.add_future_labels(data, horizons=1)
        self.assertEqual(labeled.loc[0, "label_day1"], "Good")
        self.assertTrue(pd.isna(labeled.loc[1, "label_day1"]))
        self.assertEqual(labeled.loc[2, "label_day1"], "Unhealthy")

    def test_chronological_split_purges_all_future_target_rows(self):
        split_path = ROOT / "scripts" / "04_train_test_split.py"
        spec = importlib.util.spec_from_file_location("train_test_split_script", split_path)
        split_script = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(split_script)

        frame = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "label_day3": np.arange(100),
        })
        train, test = split_script.chronological_purged_split(frame, load_experiment_config())
        horizon_gap = load_experiment_config()["forecast_horizon_rows"]
        self.assertLess(train["date"].max() + pd.Timedelta(days=horizon_gap), test["date"].min())

    def test_cleaning_imputation_never_uses_future_values(self):
        path = ROOT / "scripts" / "01_data_cleaning.py"
        spec = importlib.util.spec_from_file_location("cleaning_script", path)
        cleaner = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(cleaner)
        data = observations(5)[model_contract.OBSERVATION_COLUMNS].copy()
        data.loc[2, model_contract.OBSERVATION_COLUMNS] = np.nan
        original_fill = cleaner.causally_fill_measurements(data)
        changed = data.copy()
        changed.loc[4, "pm25"] = 900
        changed_fill = cleaner.causally_fill_measurements(changed)
        self.assertEqual(original_fill.loc[2, "pm25"], changed_fill.loc[2, "pm25"])


    def test_versioned_artifact_round_trip_and_rejects_mismatched_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_day1.pkl"
            estimator = FakeEstimator()
            model_contract.save_model_artifact(estimator, path, horizon=1)
            artifact = model_contract.load_model_artifact(path)
            self.assertIs(artifact["estimator"].__class__, FakeEstimator)
            self.assertEqual(artifact["feature_names"], model_contract.FEATURE_COLUMNS)

            incompatible = model_contract.make_model_artifact(estimator, horizon=1)
            incompatible["feature_names"] = incompatible["feature_names"][:-1]
            joblib.dump(incompatible, path)
            with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
                model_contract.load_model_artifact(path)

    def test_api_uses_shared_schema_and_keeps_forecast_response_shape(self):
        edge = load_edge_module()
        estimator = FakeEstimator()
        base_history = observations(5)
        edge.history_df = base_history.copy()
        edge.models = {f"day{horizon}": estimator for horizon in range(1, 4)}
        edge.model_errors = {}
        today = datetime.now()
        weather = {
            (today + timedelta(days=horizon)).strftime("%Y-%m-%d"): {
                "temperature_2m": 25.0,
                "relative_humidity_2m": 60.0,
                "surface_pressure": 1002.0,
                "windspeed_10m": 3.0,
                "winddirection_10m": 80.0,
            }
            for horizon in (2, 3)
        }

        with tempfile.TemporaryDirectory() as directory:
            edge.HISTORY_LOG_PATH = Path(directory) / "history.csv"
            client = edge.app.test_client()
            with patch.object(edge, "fetch_forecast_weather", return_value=weather):
                response = client.post("/predict", json={
                    "pm25": 32.0,
                    "temperature": 26.0,
                    "humidity": 58.0,
                    "pressure": 1001.0,
                    "windspeed": 3.2,
                    "winddirection": 90.0,
                })

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(set(body["forecast"]), {"day1", "day2", "day3"})
        self.assertEqual(body["forecast"]["day1"]["predicted_category"], "Moderate")
        self.assertEqual(body["input_received"]["pm25"], 32.0)
        self.assertEqual(estimator.last_features.columns.tolist(), model_contract.FEATURE_COLUMNS)
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["status"], "ok")
        self.assertEqual(set(health.get_json()["models_loaded"]), {"day1", "day2", "day3"})

    def test_offline_demo_runs_end_to_end_and_writes_provenance(self):
        path = ROOT / "scripts" / "run_demo.py"
        spec = importlib.util.spec_from_file_location("run_demo_script", path)
        demo = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(demo)
        with tempfile.TemporaryDirectory() as directory:
            demo.REPORT_PATH = Path(directory) / "demo_report.json"
            report = demo.run_demo()
            self.assertTrue(report["demo_only"])
            self.assertEqual(set(report["horizons"]), {"day1", "day2", "day3"})
            self.assertEqual(report["purge_gap_rows"], 3)
            self.assertTrue(demo.REPORT_PATH.exists())


if __name__ == "__main__":
    unittest.main()
