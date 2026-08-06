# UCCpHTP — CFD Pressure Coefficient Surrogate for Horizontal Tail Plane

## Overview

UCCpHTP trains a surrogate model that predicts the pressure coefficient **Cp** at any point on a Horizontal Tail Plane (HTP) surface given its 3D coordinates and the aerodynamic flight condition. The training data comes from high-fidelity CFD simulations (~8.2M points), making a full neural-network surrogate the practical alternative to storing and querying the full CFD field.

---

## Objective

Build an MLP surrogate that approximates the CFD pressure field:

```
f(x, y, z, alpha, mach) → Cp
```

Achieving Q90 relative error < 10 % on Cp, enabling rapid pressure-field reconstruction at arbitrary conditions without re-running the CFD solver.

---

## Dataset

| Property | Value |
|---|---|
| Source | CFD simulation (PyLOM solver) on HTP surface mesh |
| Format | Pre-split CSV files in `data/` (see below) |
| Rows | ~8.2 M CFD surface points |
| Split | Pre-split — provided directly as train / val / test files |

> **Note:** The CFD dataset is provided as pre-split files. Place them in `UCCpHTP/data/` before running:
>
> | File | Contents |
> |---|---|
> | `x_train.csv`, `x_val.csv`, `x_test.csv` | Input columns: `x, y, z, alpha, mach` |
> | `yt_train.csv`, `yt_val.csv`, `yt_test.csv` | True output column: `Cp` |
> | `yh_train.csv`, `yh_val.csv`, `yh_test.csv` | External model predictions (optional, not used in training) |

### Inputs (5 features)

| Variable | Type | Description |
|---|---|---|
| `x` | Continuous | X coordinate on HTP surface (m) |
| `y` | Continuous | Y coordinate on HTP surface (m) |
| `z` | Continuous | Z coordinate on HTP surface (m) |
| `alpha` | Continuous | Angle of attack (deg) |
| `mach` | Continuous | Mach number |

### Outputs (1 target)

| Variable | Description |
|---|---|
| `Cp` | Pressure coefficient at the surface point |

---

## Pipeline Architecture

| Stage | Notebook | Description |
|---|---|---|
| SF_1 | `SF_1_Requirements.ipynb` | Define accuracy targets (Q90 < 0.10 for Cp) |
| SF_2 | `SF_2_Data_Acquisition.ipynb` | Load `cphtp_data.csv`; log dataset statistics |
| SF_3 | `SF_3_Data_Cleaning.ipynb` | Drop rows with missing values |
| SF_4 | `SF_4_Data_Partitioning.ipynb` | 70 / 10 / 20 % train / val / test split |
| SF_5 | `SF_5_Feature_Selection.ipynb` | Fit `normalizer_transformer` (MinMax) on training set |
| SF_6 | `SF_6_Model_Selection.ipynb` | Define MLP architecture and hyperparameters |
| SF_7 | `SF_7_Model_Training.ipynb` | Train MLP; log loss curve to MLflow |
| SF_8 | `SF_8_Model_Deployment.ipynb` | Package scaler + model into sklearn Pipeline (.pkl) |
| SF_9 | `SF_9_Model_Validation.ipynb` | Metrics, distribution tests, scatter/ratio plots, HTML report |

---

## Model

### Recommended: MLP Regressor

The output Cp is a smooth, continuous field over the HTP surface. The original UCCpHTP pipeline used a PyTorch MLP (4 hidden layers, 252 neurons per layer, dropout=0.2). This SF 2.2 implementation replicates that architecture with sklearn's `MLPRegressor` and early stopping. GradientBoosting is impractical at ~8.2M rows due to memory and compute constraints.

| Hyperparameter | Value |
|---|---|
| Architecture | 5 → 256 → 128 → 64 → 1 |
| Activation | ReLU |
| Solver | Adam |
| L2 regularisation (α) | 0.0001 |
| Batch size | 512 |
| Learning rate | Adaptive (init 0.001) |
| Max iterations | 500 |
| Early stopping | Yes — patience 20 epochs, tol 1 × 10⁻⁶ |

---

## How to Run

**Data prerequisite:**
```
UCCpHTP/data/
├── x_train.csv   x_val.csv   x_test.csv    ← inputs (x, y, z, alpha, mach)
├── yt_train.csv  yt_val.csv  yt_test.csv   ← true output (Cp)
└── yh_train.csv  yh_val.csv  yh_test.csv   ← optional, not used in training
```

**Option A — Elyra visual editor (recommended):**
```bash
~/Desktop/Pipelines/start_jupyter.sh
# Open UCCpHTP.pipeline → Run Pipeline (local)
```

**Option B — standalone script:**
```bash
cd ~/Desktop/Pipelines
source .venv/bin/activate
MLFLOW_ALLOW_FILE_STORE=true python UCCpHTP/pipeline/run_pipeline.py
```

**Option C — notebook by notebook:**
Open each `SF_N_*.ipynb` in order.

**Production run (more iterations):**
```bash
# Edit SF_6_Model_Selection.yaml: max_iter: 500 → max_iter: 2000
# Delete pipeline/data/ folder
MLFLOW_ALLOW_FILE_STORE=true python UCCpHTP/pipeline/run_pipeline.py
```

---

## Results

> Results pending — requires the CFD dataset to be provided.

| Output | Requirement | Status |
|---|---|---|
| Cp | Q90 < 0.10 | Awaiting data |

The original PyLOM-based pipeline achieved strong Cp reconstruction on the HTP surface. This SF 2.2 port is expected to reach similar accuracy once the full CFD dataset is provided and max_iter is increased for the production run.

---

## Repository Structure

```
UCCpHTP/
├── README.md
├── data/
│   └── cphtp_data.csv                  ← CFD dataset (not included — provide before running)
└── pipeline/
    ├── pipeline_config.yaml
    ├── UCCpHTP.pipeline                ← Elyra visual pipeline graph
    ├── run_pipeline.py                 ← standalone runner (exits with error if data missing)
    ├── SF_1_Requirements.ipynb
    ├── SF_2_Data_Acquisition.ipynb
    ├── SF_3_Data_Cleaning.ipynb
    ├── SF_4_Data_Partitioning.ipynb
    ├── SF_5_Feature_Selection.ipynb
    ├── SF_6_Model_Selection.ipynb
    ├── SF_7_Model_Training.ipynb
    ├── SF_8_Model_Deployment.ipynb
    ├── SF_9_Model_Validation.ipynb
    ├── metadata/                       ← SF_1…SF_9 YAML configuration files
    └── python_nodes_library/           ← @sf.node functions
        ├── data_acquisition/
        ├── data_cleansing/
        ├── feature_selection/
        ├── model_training/
        ├── model_deployment/
        └── model_validation/
```
