"""Train and register the v1 MP257 water-demand model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
NUMERIC_FEATURES = (
    "flats_occupied",
    "temp_c",
    "garden_area_m2",
    "tank_level_morning_pct",
    "leak_complaints",
    "supply_hours",
)
CATEGORICAL_FEATURES = ("supply_source", "tower")


def load_params(params_path: str | Path | None = None) -> dict[str, Any]:
    """Load project settings from params.yaml."""
    path = Path(params_path) if params_path else ROOT / "params.yaml"
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)
    if not isinstance(params, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return params


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _make_estimator(kind: str, params: dict[str, Any], seed: int):
    model_params = dict(params)
    model_params.setdefault("random_state", seed)
    if kind == "random_forest":
        model_params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**model_params)
    if kind == "extra_trees":
        model_params.setdefault("n_jobs", -1)
        return ExtraTreesRegressor(**model_params)
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(**model_params)
    raise ValueError(f"Unsupported estimator type: {kind}")


def build_pipeline(kind: str, params: dict[str, Any], seed: int) -> Pipeline:
    """Build preprocessing and regression as one serializable pipeline."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("regressor", _make_estimator(kind, params, seed)),
        ]
    )


def _load_model_frame(
    params: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], str, str]:
    training = params["training"]
    clean_path = _project_path(params["paths"]["cleaned_train"])
    original_path = _project_path(params["paths"]["original_train"])
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Cleaned training data not found at {clean_path}. "
            "Run the data-cleaning step after its input paths are corrected."
        )
    frame = pd.read_csv(clean_path)
    features = list(training["features"])
    target = training["target"]
    missing = sorted(set(features + [target]) - set(frame.columns))
    if missing:
        raise ValueError(f"Cleaned training data is missing required columns: {missing}")
    if not original_path.exists():
        raise FileNotFoundError(f"Original training data not found at {original_path}")
    sha256 = hashlib.sha256(original_path.read_bytes()).hexdigest()
    return frame, features, target, sha256


def train_v1(params_path: str | Path | None = None) -> dict[str, Any]:
    """Compare three configured models, fit the winner on all clean train data,
    log the runs to MLflow, and register v1 with the champion alias.
    """
    params = load_params(params_path)
    training = params["training"]
    frame, features, target, data_sha256 = _load_model_frame(params)
    x = frame[features]
    y = frame[target]
    seed = int(params["project"]["random_seed"])
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=float(training["validation_size"]),
        random_state=seed,
    )

    mlflow_cfg = params["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    client = MlflowClient()
    model_name = mlflow_cfg["registered_model_name"]
    try:
        existing = client.get_registered_model(model_name)
    except MlflowException:
        existing = None
    if existing is not None:
        raise RuntimeError(
            f"Registered model {model_name!r} already exists. "
            "Check its versions before creating v1 again."
        )

    candidates = training["candidates"]
    candidate_results: list[dict[str, Any]] = []
    for name, spec in candidates.items():
        model = build_pipeline(spec["estimator"], spec.get("params", {}), seed)
        with mlflow.start_run(run_name=f"v1_candidate_{name}") as run:
            mlflow.log_params(
                {
                    "model_name": name,
                    "data_sha256": data_sha256,
                    "random_seed": seed,
                    "validation_size": float(training["validation_size"]),
                    **{
                        f"model.{key}": json.dumps(value, sort_keys=True)
                        if value is None or isinstance(value, (dict, list))
                        else value
                        for key, value in spec.get("params", {}).items()
                    },
                }
            )
            model.fit(x_train, y_train)
            validation_mae = float(mean_absolute_error(y_valid, model.predict(x_valid)))
            mlflow.log_metric("mae_validation", validation_mae)
            candidate_results.append(
                {
                    "name": name,
                    "estimator": spec["estimator"],
                    "params": dict(spec.get("params", {})),
                    "validation_mae": validation_mae,
                    "run_id": run.info.run_id,
                }
            )

    best = min(candidate_results, key=lambda result: result["validation_mae"])
    final_model = build_pipeline(best["estimator"], best["params"], seed)
    final_model.fit(x, y)

    with mlflow.start_run(run_name="v1_registered") as run:
        mlflow.log_params(
            {
                "model_name": best["name"],
                "selected_estimator": best["estimator"],
                "data_sha256": data_sha256,
                "random_seed": seed,
                "training_rows": len(frame),
                "selected_from_run": best["run_id"],
                **{
                    f"selected_param_{key}": json.dumps(value, sort_keys=True)
                    for key, value in best["params"].items()
                },
            }
        )
        mlflow.log_metric("mae_validation", best["validation_mae"])
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            serialization_format="cloudpickle",
            input_example=x.head(5),
        )
        version = mlflow.register_model(model_info.model_uri, model_name)
        client.set_registered_model_alias(model_name, "champion", version.version)

    model_path = _project_path(params["paths"]["local_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)

    result = {
        "selected_model": best["name"],
        "validation_mae": best["validation_mae"],
        "best_candidate_run_id": best["run_id"],
        "best_run_id": run.info.run_id,
        "v1_run_id": run.info.run_id,
        "experiment_name": mlflow_cfg["experiment_name"],
        "registered_model_name": model_name,
        "v1_version": version.version,
        "champion_alias": "champion",
        "local_model_path": str(model_path),
        "data_sha256": data_sha256,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    train_v1()
