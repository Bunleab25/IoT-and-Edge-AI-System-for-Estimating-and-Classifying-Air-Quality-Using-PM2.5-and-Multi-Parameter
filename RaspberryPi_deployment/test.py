"""Print the validated feature schema for each deployed horizon model."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from aerosense.model_contract import load_model_artifact  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"

for horizon in range(1, 4):
    artifact_path = MODELS_DIR / f"best_day{horizon}.pkl"
    artifact = load_model_artifact(artifact_path)
    print(f"Day {horizon}: {len(artifact['feature_names'])} features")
    for feature in artifact["feature_names"]:
        print(f" - {feature}")
