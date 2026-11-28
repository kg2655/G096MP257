# SCSE3040 MLOps Mini Project — MP257

## Project

| Field | Details |
|---|---|
| Group | G096 |
| Problem | MP257 — Housing Society Water Demand (Surat) |
| Batch | 61 |
| Faculty | Gaurav Tripathi |

The project builds an MLOps pipeline to predict a housing society's daily water use in litres. This is a regression task, evaluated with mean absolute error (MAE).

## Team

| Enrollment number | Member |
|---|---|
| S24CSEU2020 | Rahul Rautela |
| S24CSEU2595 | Kannu Goyal |
| S24CSEU0780 | Garvit Maheshwari |

## Data

| File | Rows | Dates | Purpose |
|---|---:|---|---|
| `data/raw/train.csv` | 1,000 | Jan–Jun 2026 | Training data; includes `litres_per_day` target |
| `data/raw/later.csv` | 300 | Jul–Aug 2026 | Monitoring and retraining data; includes target |
| `data/raw/holdout_features.csv` | 250 | Sep 2026 | Final prediction inputs; target is not provided |

The DVC metadata for `train.csv` and `later.csv` is versioned in Git. The CSV files are excluded from Git. No shared DVC storage remote is configured, so each group member must obtain the supplied dataset files separately and place them at the paths above before working with the data.

## Local setup

Use Windows PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the supplied CSV files in `data/raw/`. Verify the tracked data after placing the files with:

```powershell
.\.venv\Scripts\dvc.exe status
```

The pinned project dependencies are listed in `requirements.txt`. API-serving commands will be added when the service is integrated.

## Run the data and modeling pipeline

Run these commands from the repository root, in order, after placing all three supplied CSV files in `data/raw/` and installing the requirements above:

```powershell
.\.venv\Scripts\python.exe src\water_demand\data_checks.py
.\.venv\Scripts\python.exe src\water_demand\modeling.py
.\.venv\Scripts\python.exe src\water_demand\monitoring.py
```

The first command checks the seven data issues, cleans the training data, and updates the data section of `findings.yaml`. The second compares the configured v1 candidates, logs their results to MLflow, registers the best model as v1, and sets the `champion` alias. The third calculates drift and tower-level MAE, trains and evaluates v2, applies the promotion rule, updates `findings.yaml`, and writes `predictions/holdout_predictions.csv` using the champion model.

These commands create MLflow runs and registered model versions. The repository already contains the generated results and a populated `mlflow.db`; do not rerun the modeling and monitoring commands against that database unless you intend to create new model versions and refresh the recorded findings and predictions. The v1 training command stops if the registered model already exists.
