# Surrogate Factory — MLOps Pipelines for Aerospace Engineering

## Overview

This repository contains five surrogate modelling use cases built on **Surrogate Factory v2.2**, an Airbus Inner Source MLOps framework that structures the development of physics-informed machine learning models into 9 reproducible pipeline stages. Each use case replaces expensive simulation workflows (CFD, FEM, structural analysis) with fast surrogate models validated to engineering accuracy standards.

---

## What is Surrogate Factory?

Surrogate Factory (SF) is an MLOps framework designed for aerospace engineering teams. It enforces a structured, auditable workflow from requirements to validated deployment, wrapping standard ML algorithms (sklearn, PyTorch) in a traceable pipeline.

### 9-Stage Pipeline

| Stage | Name | What it does |
|---|---|---|
| SF_1 | Requirements | Define quantitative accuracy targets (e.g. Q90 < 10 %) |
| SF_2 | Data Acquisition | Load and parse raw simulation data |
| SF_3 | Data Cleansing | Handle missing values and outliers |
| SF_4 | Data Partitioning | 70 / 10 / 20 % train / val / test split |
| SF_5 | Feature Selection | Fit preprocessors (StandardScaler, MinMax, OHE) |
| SF_6 | Model Selection | Define algorithm architectures and hyperparameters in YAML |
| SF_7 | Model Training | Train all candidate models; log metrics to MLflow |
| SF_8 | Model Deployment | Package scaler + model as an sklearn Pipeline (.pkl) |
| SF_9 | Model Validation | Metrics, distribution tests, scatter/ratio plots, HTML report |

### Key components

| Component | Description |
|---|---|
| `@sf.node` | Decorator that wraps Python functions into traceable pipeline nodes |
| `Workflow` | Central object that loads config, manages metadata, and saves/loads data and artifacts |
| `pipeline_config.yaml` | Per-UC configuration: paths, MLflow tracker, algorithm catalog |
| `SF_N_*.yaml` | One YAML file per stage — defines inputs, outputs, model hyperparameters |
| `{UC}.pipeline` | Elyra visual pipeline graph linking the 9 notebooks |
| MLflow | Experiment tracker — logs loss curves, metrics, and model artifacts |
| `validationlib` | Airbus Inner Source validation library — split quality, bias detection, uncertainty models, HTML reports |

---

## Repository Structure

```
Pipelines/
├── README.md                   ← this file
├── src/
│   └── surrogate_factory/      ← SF 2.2 source package (installed with pip install -e .)
├── validationlib/              ← Airbus Inner Source validation library
├── start_jupyter.sh            ← launch JupyterLab with the shared .venv
├── pyproject.toml              ← package definition for surrogate_factory
│
├── UCAirfoils/                 ← aerodynamic surrogate for airfoil CST design space
├── UCCpHTP/                    ← CFD pressure coefficient surrogate on HTP surface
├── UCFatigue/                  ← fatigue load surrogate (FEM → MLP/GB)
├── UCHardLanding/              ← structural load surrogate for hard landing analysis
└── UCLoads/                    ← structural loads from CFD .mon files (5-model comparison)
```

---

## Use Cases Summary

| Use Case | Domain | Inputs | Outputs | Model | Data size |
|---|---|---|---|---|---|
| [UCAirfoils](UCAirfoils/README.md) | Aerodynamics | α, Re, 18 Kulfan CST coefficients | Cl, Cd, Cm | MLP 128→64→32 | 7 987 rows (NeuralFoil) |
| [UCCpHTP](UCCpHTP/README.md) | CFD / Aerodynamics | x, y, z, alpha, mach | Cp | MLP 256→128→64 | ~8.2M rows (CFD) |
| [UCFatigue](UCFatigue/README.md) | Structural / Fatigue | 8 flight params (categorical + continuous) | 7 fatigue loads | GradientBoosting + MLP | ~870 rows (FEM subset) |
| [UCHardLanding](UCHardLanding/README.md) | Structural | 7 landing params (6 discrete + 1 continuous) | O1, O2 | GradientBoosting + MLP | 89 357 rows |
| [UCLoads](UCLoads/README.md) | Structural / Aeroloads | Mach, Altitude, ALPHA | 8 structural loads (2 stations) | PyTorchNN + GB + RF + XGB + MLP | 2 000 rows synthetic (357k real) |

---

## What We Found

### Model selection insights

- **Discrete inputs → GradientBoosting**: UCHardLanding (6 of 7 inputs are discrete) and UCFatigue (2 categorical inputs after OHE) both benefit from tree-based models that split naturally at level boundaries.
- **Smooth continuous outputs → MLP**: UCAirfoils and UCCpHTP have purely continuous inputs and smooth output manifolds (aerodynamic coefficients, pressure fields) — MLPs outperform tree models here.
- **Large datasets + deep learning**: UCLoads shows that a residual PyTorchNN (4 ResBlocks, 128 hidden units) achieves the best R² and zero overfitting on all 8 outputs when trained with Y-standardisation.

### Validation findings

| Use Case | Best model | Key result |
|---|---|---|
| UCAirfoils | MLP | Run SF_9 on the 7 987-row dataset to evaluate |
| UCCpHTP | MLP | Pending CFD dataset — expected to match original PyLOM performance |
| UCFatigue | GradientBoosting | R² ≥ 0.929 on 5/7 outputs; Frontal gust and Giro fail Q90 (sparse data) |
| UCHardLanding | GradientBoosting | Expected to comfortably meet Q90 < 0.10 with 89k rows + discrete structure |
| UCLoads | PyTorchNN | Best R² on 7/8 outputs; KS 8/8 ✓ (no overfitting); Q90 fails on synthetic noise (expected) |

### Recurring issues and fixes

| Issue | Fix |
|---|---|
| `mlflow.exceptions.MlflowException: filesystem tracking backend in maintenance mode` | Run with `MLFLOW_ALLOW_FILE_STORE=true python run_pipeline.py` |
| GitHub 100 MB file limit (RandomForest .modl files are 1.6 GB) | Exclude via `.gitignore`; regenerate locally after cloning |
| NeuralFoil `.item()` vs `float()` on numpy arrays (NumPy ≥ 1.25) | Use `.item()` on all NeuralFoil result arrays |
| UCCpHTP data not available locally | Pipeline exits with a clear error message if `cphtp_data.csv` is missing |

---

## Quick Start

### Environment setup

```bash
git clone <repo-url>
cd Pipelines
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .                        # installs surrogate_factory
pip install jupyterlab elyra openpyxl scipy ipykernel sympy statsmodels xgboost torch neuralfoil
```

### Run any pipeline

**Option A — Elyra visual editor:**
```bash
./start_jupyter.sh
# Open {UC}/{UC}.pipeline → Run Pipeline (local)
```

**Option B — standalone script:**
```bash
source .venv/bin/activate
MLFLOW_ALLOW_FILE_STORE=true python {UC}/pipeline/run_pipeline.py
```

---

## Source Packages

| Package | Location | Description |
|---|---|---|
| `surrogate_factory` | `src/surrogate_factory/` | SF 2.2 framework — Workflow, @sf.node, catalog (MLPRegressor, normalizer_transformer, metrics) |
| `validationlib` | `validationlib/` | Airbus validation library — VTP split quality, KS/AD/MW tests, bias detection, uncertainty models, HTML report template |

Both packages are imported directly from the repository (no PyPI install needed):
- `surrogate_factory`: installed via `pip install -e .` (editable mode)
- `validationlib`: added to `sys.path` in each `run_pipeline.py`
