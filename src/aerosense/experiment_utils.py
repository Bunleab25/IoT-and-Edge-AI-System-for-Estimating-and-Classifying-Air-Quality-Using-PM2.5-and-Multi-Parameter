"""Helpers for experiment configuration and reproducibility manifests."""
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/experiment.json"
CONFIG_SCHEMA_VERSION = 1
PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib"]
SOURCE_FILES = [
    "scripts/01_data_cleaning.py",
    "scripts/03_feature_engineering.py",
    "scripts/04_train_test_split.py",
    "scripts/05_train_model.py",
    "scripts/06_evaluate.py",
    "scripts/_bootstrap.py",
    "pyproject.toml",
    "src/aerosense/evaluation_metrics.py",
    "src/aerosense/experiment_utils.py",
    "src/aerosense/model_contract.py",
    "scripts/run_demo.py",
]


def load_experiment_config(path: Path = CONFIG_PATH) -> dict:
    """Load and validate the JSON experiment configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment config schema: {config.get('schema_version')!r}")
    if not isinstance(config.get("seed"), int) or config["seed"] < 0:
        raise ValueError("Experiment seed must be a non-negative integer.")
    if not 0 < config.get("train_fraction", 0) < 1:
        raise ValueError("train_fraction must be between zero and one.")
    if config.get("forecast_horizon_rows") != 3:
        raise ValueError("forecast_horizon_rows must match the three-day target contract.")
    if not isinstance(config.get("time_series_cv_splits"), int) or config["time_series_cv_splits"] < 2:
        raise ValueError("time_series_cv_splits must be an integer of at least two.")
    if not isinstance(config.get("search_iterations"), int) or config["search_iterations"] < 1:
        raise ValueError("search_iterations must be a positive integer.")
    if config.get("selection_metric") != "f1_macro":
        raise ValueError("Only f1_macro is supported for model selection.")
    for model_name in ("xgboost", "lightgbm"):
        parameters = config.get(model_name, {}).get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(f"Missing parameter distributions for {model_name}.")
    if config.get("classes") != ["Very Good", "Good", "Moderate", "Unhealthy"]:
        raise ValueError("Experiment classes must match the model label contract.")
    return config


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    """Return installed versions for experiment dependencies."""
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def git_revision(project_root: Path = PROJECT_ROOT) -> str | None:
    """Return the checked-out Git revision when available."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_worktree_dirty(project_root: Path = PROJECT_ROOT) -> bool | None:
    """Report whether tracked or untracked working-tree changes exist."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def source_fingerprints(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Hash the pipeline sources that define a training or demo run."""
    return {
        relative_path: sha256_file(project_root / relative_path)
        for relative_path in SOURCE_FILES
        if (project_root / relative_path).exists()
    }


def build_training_manifest(
    config: dict,
    config_path: Path,
    train_path: Path,
    test_path: Path,
    feature_contract_path: Path,
    split_summary: dict,
    selected_models: dict,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """Create a provenance record for the exact training data and settings."""
    return {
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(project_root),
        "git_worktree_dirty": git_worktree_dirty(project_root),
        "source_sha256": source_fingerprints(project_root),
        "config": config,
        "config_sha256": sha256_file(config_path),
        "inputs": {
            "train_csv": {"path": str(train_path.relative_to(project_root)), "sha256": sha256_file(train_path)},
            "test_csv": {"path": str(test_path.relative_to(project_root)), "sha256": sha256_file(test_path)},
            "feature_contract": {
                "path": str(feature_contract_path.relative_to(project_root)),
                "sha256": sha256_file(feature_contract_path),
            },
        },
        "split": split_summary,
        "packages": package_versions(),
        "selected_models": selected_models,
    }


def write_json(path: Path, payload: dict) -> None:
    """Write indented UTF-8 JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
