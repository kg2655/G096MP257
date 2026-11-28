"""Monitor v1, train v2, apply the promotion rule, and score the holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from modeling import (
    NUMERIC_FEATURES,
    ROOT,
    _project_path,
    build_pipeline,
    load_params,
)


TOWER_LEVELS = ("Tower A", "Tower B", "Tower C")


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """Compute PSI using reference quantile bins and a separate missing bin."""
    reference_values = pd.to_numeric(reference, errors="coerce")
    current_values = pd.to_numeric(current, errors="coerce")
    reference_nonmissing = reference_values.dropna().to_numpy(dtype=float)
    current_nonmissing = current_values.dropna().to_numpy(dtype=float)
    if reference_nonmissing.size == 0 or current_nonmissing.size == 0:
        raise ValueError("PSI requires at least one non-missing value in both samples")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference_nonmissing, quantiles))
    if edges.size == 1:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    reference_counts = np.histogram(reference_nonmissing, bins=edges)[0].astype(float)
    current_counts = np.histogram(current_nonmissing, bins=edges)[0].astype(float)
    reference_counts = np.append(reference_counts, reference_values.isna().sum())
    current_counts = np.append(current_counts, current_values.isna().sum())

    # Small additive smoothing keeps PSI finite when a bin is empty.
    epsilon = 1e-6
    reference_share = (reference_counts + epsilon) / (
        reference_counts.sum() + epsilon * reference_counts.size
    )
    current_share = (current_counts + epsilon) / (
        current_counts.sum() + epsilon * current_counts.size
    )
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def _selected_v1_config(client: MlflowClient, model_name: str) -> tuple[str, dict[str, Any], str]:
    version = client.get_model_version(model_name, "1")
    run = client.get_run(version.run_id)
    logged = run.data.params
    estimator = logged["selected_estimator"]
    model_params = {
        key.removeprefix("selected_param_"): json.loads(value)
        for key, value in logged.items()
        if key.startswith("selected_param_")
    }
    return estimator, model_params, version.run_id


def _update_findings(
    findings_path: Path,
    results: dict[str, Any],
    experiment_name: str,
    model_name: str,
    v1_run_id: str,
    champion_version: int,
) -> None:
    """Update monitoring and tracking values while preserving the other sections."""
    with findings_path.open(encoding="utf-8") as file:
        findings = yaml.safe_load(file)
    if not isinstance(findings, dict) or not {"tracking", "monitoring"}.issubset(findings):
        raise ValueError(f"Expected tracking and monitoring sections in {findings_path}")

    tracking = findings["tracking"]
    monitoring = findings["monitoring"]
    expected_psi = monitoring["psi"]
    expected_groups = monitoring["mae_by_group"]
    if not set(results["psi"]).issubset(expected_psi):
        raise ValueError("findings.yaml PSI keys do not match the configured numeric features")
    if not set(results["mae_by_group"]).issubset(expected_groups):
        raise ValueError("findings.yaml tower keys do not match the later dataset")

    tracking["experiment_name"] = experiment_name
    tracking["best_run_id"] = v1_run_id
    tracking["registered_model_name"] = model_name
    tracking["champion_version"] = champion_version

    for feature, value in results["psi"].items():
        expected_psi[feature] = value
    monitoring["most_drifted_feature"] = results["most_drifted_feature"]
    for tower, value in results["mae_by_group"].items():
        expected_groups[tower] = value
    monitoring["worst_served_group"] = results["worst_served_group"]
    monitoring["mae_v1_on_check"] = results["mae_v1_on_check"]
    monitoring["mae_v2_on_check"] = results["mae_v2_on_check"]
    monitoring["decision"] = results["decision"]

    findings_path.write_text(
        yaml.safe_dump(findings, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _validate_findings_template(findings_path: Path) -> None:
    """Fail early if data cleaning changed the marker's required field names."""
    with findings_path.open(encoding="utf-8") as file:
        findings = yaml.safe_load(file)
    required_data_fields = {
        "leaking_column": {"present", "column"},
        "unit_mixup": {"present", "column", "group_value", "factor"},
        "sentinel_value": {"present", "column", "value", "count"},
        "duplicate_rows": {"present", "count"},
        "stuck_sensor": {"present", "column", "start_date", "end_date"},
        "target_typos": {"present", "count"},
        "spelling_variants": {"present", "column", "true_levels"},
    }
    data_issues = findings.get("data_issues", {}) if isinstance(findings, dict) else {}
    for issue, required in required_data_fields.items():
        actual = data_issues.get(issue, {})
        missing = required - set(actual)
        if missing:
            raise ValueError(
                f"findings.yaml data_issues.{issue} is missing required fields: "
                f"{sorted(missing)}. Restore the original marker template keys first."
            )


def run_monitoring_and_v2(params_path: str | Path | None = None) -> dict[str, Any]:
    """Calculate PSI and v1 errors, train/evaluate v2, and predict the holdout."""
    params = load_params(params_path)
    paths = params["paths"]
    training = params["training"]
    mlflow_cfg = params["mlflow"]
    findings_path = ROOT / "findings.yaml"
    _validate_findings_template(findings_path)
    seed = int(params["project"]["random_seed"])
    features = list(training["features"])
    target = training["target"]

    clean_train_path = _project_path(paths["cleaned_train"])
    original_train_path = _project_path(paths["original_train"])
    later_path = _project_path(paths["later"])
    holdout_path = _project_path(paths["holdout"])
    for path in (clean_train_path, original_train_path, later_path, holdout_path):
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    clean_train = pd.read_csv(clean_train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)
    for label, frame, required in (
        ("cleaned training", clean_train, features + [target]),
        ("later", later, features + [target, "date"]),
        ("holdout", holdout, features + ["record_id"]),
    ):
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{label} data is missing required columns: {missing}")

    data_sha256 = hashlib.sha256(original_train_path.read_bytes()).hexdigest()
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    client = MlflowClient()
    model_name = mlflow_cfg["registered_model_name"]
    v1_run_id = client.get_model_version(model_name, "1").run_id
    v1_model = mlflow.sklearn.load_model(f"models:/{model_name}/1")

    psi_values = {
        feature: population_stability_index(clean_train[feature], later[feature], bins=10)
        for feature in NUMERIC_FEATURES
    }
    later_predictions = v1_model.predict(later[features])
    tower_mae = {
        tower: float(
            mean_absolute_error(
                later.loc[later["tower"] == tower, target],
                np.asarray(later_predictions)[later["tower"].eq(tower).to_numpy()],
            )
        )
        for tower in TOWER_LEVELS
        if (later["tower"] == tower).any()
    }
    missing_towers = set(TOWER_LEVELS) - set(tower_mae)
    if missing_towers:
        raise ValueError(f"later.csv is missing required tower values: {sorted(missing_towers)}")

    sorted_later = later.sort_values("date", kind="mergesort").reset_index(drop=True)
    split_at = int(len(sorted_later) * 0.70)
    if split_at == 0 or split_at >= len(sorted_later):
        raise ValueError("later.csv needs enough rows for a 70/30 date-ordered split")
    later_train = sorted_later.iloc[:split_at]
    later_check = sorted_later.iloc[split_at:]

    v1_check_predictions = v1_model.predict(later_check[features])
    mae_v1_check = float(mean_absolute_error(later_check[target], v1_check_predictions))

    estimator, model_params, _ = _selected_v1_config(client, model_name)
    v2_training = pd.concat([clean_train, later_train], ignore_index=True)
    v2_model = build_pipeline(estimator, model_params, seed)
    v2_model.fit(v2_training[features], v2_training[target])
    v2_check_predictions = v2_model.predict(later_check[features])
    mae_v2_check = float(mean_absolute_error(later_check[target], v2_check_predictions))

    promote = mae_v2_check <= mae_v1_check * 0.88
    decision = "promote" if promote else "keep"

    with mlflow.start_run(run_name="v1_monitoring"):
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("later_rows", len(later))
        for feature, value in psi_values.items():
            mlflow.log_metric(f"psi_{feature}", value)
        for tower, value in tower_mae.items():
            mlflow.log_metric(f"mae_{tower.lower().replace(' ', '_')}", value)
        mlflow.log_metric("mae_v1_on_check", mae_v1_check)

    with mlflow.start_run(run_name="v2_retrained"):
        mlflow.log_params(
            {
                "data_sha256": data_sha256,
                "retraining_rows_from_later": len(later_train),
                "check_rows": len(later_check),
                "split_date": str(later_check["date"].iloc[0]),
                "selected_estimator": estimator,
            }
        )
        mlflow.log_metric("mae_v2_on_check", mae_v2_check)
        model_info = mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="model_v2",
            serialization_format="cloudpickle",
            input_example=v2_training[features].head(5),
        )
        v2_version = mlflow.register_model(model_info.model_uri, model_name)
        if str(v2_version.version) != "2":
            raise RuntimeError(
                f"Expected retrained model version 2, received {v2_version.version}. "
                "Inspect the registry before changing the champion alias."
            )

    if promote:
        client.set_registered_model_alias(model_name, "champion", v2_version.version)
        champion_model = v2_model
        champion_version = int(v2_version.version)
    else:
        champion_model = v1_model
        champion_version = 1

    champion_path = _project_path(paths["local_model"])
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, champion_path)

    holdout_predictions = champion_model.predict(holdout[features])
    prediction_path = _project_path(paths["holdout_predictions"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"record_id": holdout["record_id"], "prediction": holdout_predictions}
    ).to_csv(prediction_path, index=False)

    results = {
        "psi": psi_values,
        "most_drifted_feature": max(psi_values, key=psi_values.get),
        "mae_by_group": tower_mae,
        "worst_served_group": max(tower_mae, key=tower_mae.get),
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "decision": decision,
        "champion_version": champion_version,
        "prediction_file": str(prediction_path),
        "prediction_rows": len(holdout_predictions),
        "data_sha256": data_sha256,
    }
    _update_findings(
        findings_path,
        results,
        mlflow_cfg["experiment_name"],
        model_name,
        v1_run_id,
        champion_version,
    )
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_monitoring_and_v2()
