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
.\.venv\Scripts\python.exe -m pip install "dvc==3.67.1"
```

Place the supplied CSV files in `data/raw/`. Verify the tracked data after placing the files with:

```powershell
.\.venv\Scripts\dvc.exe status
```

The project's pinned application dependencies and commands for model training and API serving will be documented here as those components are added.
