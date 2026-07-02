# UCLoads — Surrogate Model for Structural Load Prediction

## Validation Results Summary (synthetic data)

> Results on **500 synthetic rows** (plausible aerodynamic physics, `random_state=42`). Real data results will differ significantly once `.mon` files from CFD are available.

### 5-model comparison (MLP · GradientBoosting · RandomForest · XGBoost · PyTorchNN)

| Output | MLP R² | GB R² | RF R² | XGB R² | PyTorchNN R² | GB Q90 | PyTorchNN Q90 | Note |
|---|---|---|---|---|---|---|---|---|
| Fz_WINGROOT | 0.922 | 0.990 | 0.975 | 0.971 | **0.994** | **0.094 ✅** | 0.142 | Vertical force at wing root |
| Mx_WINGROOT | 0.739 | 0.977 | 0.841 | 0.955 | **0.991** | 0.264 | 0.497 | Roll moment |
| My_WINGROOT | 0.917 | 0.989 | 0.974 | 0.970 | **0.992** | **0.098 ✅** | 0.133 | Pitch moment at wing root |
| Fy_FUS | 0.036 | −0.037 | −0.033 | −0.095 | 0.008 | 3.004 | 2.109 | Side force — near-zero + noise (all fail) |
| Fz_FUS | 0.929 | 0.992 | 0.977 | 0.972 | **0.993** | **0.089 ✅** | 0.163 | Vertical force at fuselage |
| Mx_FUS | 0.059 | 0.964 | 0.376 | 0.922 | **0.987** | 0.723 | 0.554 | Roll moment at fuselage |
| My_FUS | 0.928 | 0.991 | 0.980 | 0.976 | **0.993** | **0.088 ✅** | 0.161 | Pitch moment at fuselage |
| Mz_FUS | 0.838 | 0.973 | 0.895 | 0.960 | **0.989** | 0.183 | 0.342 | Yaw moment |

### KS test (overfitting indicator — should not reject H₀)

| Model | Passes / 8 outputs | Interpretation |
|---|---|---|
| **PyTorchNN** | **8 / 8** | No overfitting — train and test residuals same distribution |
| MLP | 7 / 8 | Slight generalization gap on Fz_FUS |
| GradientBoosting | 0 / 8 | Overfits (expected with 100 trees / 360 samples) |
| XGBoost | 0 / 8 | Overfits (expected with 300 trees / 360 samples) |

### Key findings

**Split quality:** excellent — residual voxel proportion = 0.0, no phacking, no isolated train points.

**PyTorchNN** (residual deep network, 4 blocks, 128 hidden units):
- Best R² on 7/8 outputs (wins over GradientBoosting on every learnable output)
- Perfect KS test: 8/8 outputs pass → no overfitting
- Early stopped at epoch 43 (val loss 0.115 in normalised space)
- **Critical**: outputs must be Y-standardised before training; without it PyTorchNN fails completely (MSE dominated by large-amplitude loads)

**RandomForest** (200 trees, `RandomForestRegressor` nativo — multi-output nativo):
- R² 0.97–0.98 en las salidas simples (Fz, My), pero falla en Mx_FUS (R²=0.38) y Mx_WINGROOT (R²=0.84)
- Sin paso Q90 — peor que GradientBoosting y XGBoost en las cargas de momento de rodadura
- KS: esperado fallo similar a GB (sobreajuste con árboles y 360 muestras)

**GradientBoosting** (100 trees per output × 8 MultiOutputRegressor):
- Best Q90: 4/8 outputs pass the ≤ 10 % target (Fz_WINGROOT, My_WINGROOT, Fz_FUS, My_FUS)
- Strong R² 0.96–0.99 on all learnable outputs
- Overfits on synthetic data (KS p < 0.05 on all 8) — regularise with `max_depth=3` on real data

**XGBoost** (300 trees per output, `nthread=1` required on macOS):
- R² 0.92–0.97 on learnable outputs, close second to GradientBoosting
- No Q90 passes (closest: My_FUS at 0.134, My_WINGROOT at 0.147)
- Overfits similarly to GradientBoosting

**MLP** (64→32 ReLU):
- R² 0.92–0.93 on simple outputs; fails on Mx_FUS (R²=0.06)
- Q90 ~0.99 on test set despite reasonable R² (predicted values off-scale on tail samples)

**Fy_FUS**: near-zero mean with high relative noise in synthetic design → all models fail this output. Not a model limitation — artefact of synthetic generation.

**Recommendation for real CFD data:**
- **PyTorchNN** is the primary candidate: best generalisation and highest accuracy
- **GradientBoosting** as reference: faster to train, interpretable, already meets Q90 on 4/8 on synthetic data
- Fy_FUS needs more training samples with varied sideslip angles

---

## Overview

UCLoads is a machine-learning surrogate that predicts aerodynamic structural loads on aircraft components from flight condition inputs. Given Mach number, altitude and angle of attack, the model predicts 8 structural load outputs across two monitor stations in milliseconds instead of running a full CFD simulation.

The pipeline is built on **Surrogate Factory v2.2**, the same MLOps framework as UCFatigue, structured into 9 reproducible stages managed through Jupyter notebooks and the Elyra visual pipeline editor.

---

## Problem Statement

Structural analysis requires evaluating aerodynamic loads at every combination of:
- Mach number (compressibility effects)
- Altitude (dynamic pressure via ISA atmosphere)
- Angle of attack (lift and moment distribution)

A surrogate trained on a representative sample of CFD/FEM simulation outputs can approximate the full mapping in real time, enabling rapid design-space sweeps and load-case screening during preliminary design.

---

## Dataset

| Property | Value |
|---|---|
| Source | `.mon` files (Tecplot monitor format from CFD solver) |
| Monitor stations | MS_WINGROOT_RHS (wing root), MS_FUS_FTJ (fuselage frame) |
| Current dataset | 500 synthetic rows (plausible physics, `random_state=42`) |
| Production dataset | Replace `TrainData/` with real CFD monitor outputs |
| Split | 70 % train / 10 % val / 20 % test (`random_state=42`) |

### Inputs (3 features — all numerical)

| Feature | Unit | Range (synthetic) | Description |
|---|---|---|---|
| Mach[-] | – | 0.30 – 0.85 | Flight Mach number |
| Altitude[ft] | ft | 0 – 40 000 | Flight altitude (ISA) |
| ALPHA[deg] | deg | −5 – 15 | Angle of attack |

### Outputs (8 structural loads)

| Output column | Raw .mon name | Station | Load component |
|---|---|---|---|
| Fz_WINGROOT | Net-Fz[N] | MS_WINGROOT_RHS | Vertical force at wing root |
| Mx_WINGROOT | Net-Mx[Nm] | MS_WINGROOT_RHS | Rolling moment at wing root |
| My_WINGROOT | Net-My[Nm] | MS_WINGROOT_RHS | Pitching moment at wing root |
| Fy_FUS | Net-Fy[N] | MS_FUS_FTJ | Lateral force at fuselage frame |
| Fz_FUS | Net-Fz[N] | MS_FUS_FTJ | Vertical force at fuselage frame |
| Mx_FUS | Net-Mx[Nm] | MS_FUS_FTJ | Rolling moment at fuselage frame |
| My_FUS | Net-My[Nm] | MS_FUS_FTJ | Pitching moment at fuselage frame |
| Mz_FUS | Net-Mz[Nm] | MS_FUS_FTJ | Yaw moment at fuselage frame |

> Column names in the raw `.mon` files (`Net-Fz[N]`, `ALPHA[deg]`, etc.) are sanitized to clean Python identifiers at load time (`Fz`, `ALPHA_deg`, etc.) to avoid conflicts with YAML parsers and statsmodels formula syntax.

---

## Pipeline Architecture

```
SF_1  Requirements          Define accuracy targets (Q90 < 10 % per output)
  ↓
SF_2  Data Acquisition      Parse .mon files; merge WINGROOT + FUS outputs by row index
  ↓
SF_3  Data Cleansing        Drop NaN rows (numerical data only — no categorical handling)
  ↓
SF_4  Data Partitioning     70 / 10 / 20 % train / val / test split
  ↓
SF_5  Feature Selection     StandardScale 3 numerical inputs → 3 scaled features
  ↓
SF_6  Model Selection       Define MLP, GradientBoosting, RandomForest, XGBoost, PyTorchNN
  ↓
SF_7  Model Training        Train all 5 models; log loss curves to MLflow
  ↓
SF_8  Model Deployment      Save sklearn Pipeline (.pkl) with scaler + model
  ↓
SF_9  Model Validation      Split quality, predictions, metrics, distribution tests,
                            requirement check, scatter/ratio plots,
                            full HTML report (validationlib template), MLflow EDA run
```

### .mon File Format (Tecplot)

```
TITLE="UCLoads Aerodynamic Monitor Data"
FILETYPE=FULL
ZONE T="monitor_run"
VARIABLES="Mach[-]""Altitude[ft]""ALPHA[deg]""Net-Fz[N]""Net-Mx[Nm]""Net-My[Nm]"
0.500  5000.000  3.000  95432.12  4521.33  -187234.50
...
```

- Lines 0–2: arbitrary header (ignored by parser)
- Line 3: `VARIABLES=` with column names separated by `""`
- Line 4+: space-separated data rows

---

## Running the Pipeline

**Option A — Elyra visual editor (recommended):**
```bash
~/Desktop/Pipelines/start_jupyter.sh
# Open UCLoads.pipeline → Run Pipeline (local)
# Kernel: UCLoads (Pipeline)
```

**Option B — standalone script:**
```bash
cd ~/Desktop/Pipelines
source .venv/bin/activate
python UCLoads/pipeline/run_pipeline.py
```

**Option C — notebook-by-notebook:**
Open each `SF_N_*.ipynb` in order using the `UCLoads (Pipeline)` kernel.

---

## Environment Setup

```bash
# From the shared .venv at Desktop/Pipelines/.venv
source .venv/bin/activate

# Register the Jupyter kernel for UCLoads notebooks
python -m ipykernel install --user --name=ucloads --display-name="UCLoads (Pipeline)"

# Launch
./start_jupyter.sh
```

> Use kernel **UCLoads (Pipeline)** in all notebooks.

---

## Models

### MLP Regressor

| Hyperparameter | Value |
|---|---|
| Architecture | 3 → 64 → 32 → 8 |
| Activation | ReLU |
| Solver | Adam |
| L2 regularisation (α) | 0.01 |
| Batch size | 32 |
| Learning rate | Adaptive (init 0.001) |
| Max iterations | 500 |
| Early stopping | Yes — patience 20 epochs, tol 1e-5 |

### Gradient Boosting

One independent `GradientBoostingRegressor` per output (8 models), wrapped in `MultiOutputRegressor`.

| Hyperparameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 4 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| min_samples_leaf | 5 |

### Random Forest

`RandomForestRegressor` — natively supports multi-output regression (no `MultiOutputRegressor` wrapper needed).

| Hyperparameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | None (fully grown) |
| min_samples_leaf | 4 |
| max_features | 1.0 (all 3 inputs) |
| n_jobs | 1 |

### XGBoost

One independent `XGBRegressor` per output (8 models), wrapped in `MultiOutputRegressor`.

| Hyperparameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 5 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 5 |
| reg_alpha / reg_lambda | 0.1 / 1.0 |
| nthread | 1 (required on macOS to avoid segfault) |

### PyTorchNN (Residual Deep Network)

Sklearn-compatible wrapper around a deep residual PyTorch network. **Requires Y-standardisation** — outputs are normalised before training and de-normalised at prediction time.

| Hyperparameter | Value |
|---|---|
| Architecture | Input → 128 → 4× ResBlock(128) → 8 |
| Block structure | Linear → LayerNorm → GELU → Linear → LayerNorm + skip + GELU |
| Optimiser | Adam (lr=0.001, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Batch size | 64 |
| Max epochs | 500 |
| Early stopping | Patience 30 (internal 15 % val split) |
| Y-normalisation | StandardScaler per output (fit inside `fit()`, applied in `predict()`) |

---

## Accuracy Requirements

All 8 outputs must satisfy **quantile 90 of relative error < 0.10** (i.e. 90 % of predictions within ±10 % of the true value).

---

## Accuracy vs UCFatigue — Key Differences

| Property | UCFatigue | UCLoads |
|---|---|---|
| Input type | Mixed (categorical + numerical) | Numerical only |
| Preprocessing | OHE + StandardScaler → 19 features | StandardScaler → 3 features |
| Outputs | 7 fatigue loads | 8 structural loads (2 stations) |
| Data source | Excel (.xlsx) | Tecplot monitor files (.mon) |
| Split | 70/10/20 | 70/10/20 |

---

## Repository Structure

```
UCLoads/
├── LOADS.md                            ← this file
├── datasets/
│   └── TrainData/
│       ├── MS_WINGROOT_RHS.mon         ← wing root loads (500 synthetic rows)
│       └── MS_FUS_FTJ.mon              ← fuselage loads (500 synthetic rows)
└── pipeline/
    ├── pipeline_config.yaml            ← paths, catalog, job config
    ├── UCLoads.pipeline                ← Elyra visual pipeline graph
    ├── run_pipeline.py                 ← standalone runner (no Jupyter)
    ├── SF_1_Requirements.ipynb
    ├── SF_2_Data_Acquisition.ipynb
    ├── SF_3_Data_Cleaning.ipynb
    ├── SF_4_Data_Partitioning.ipynb
    ├── SF_5_Feature_Selection.ipynb
    ├── SF_6_Model_Selection.ipynb
    ├── SF_7_Model_Training.ipynb
    ├── SF_8_Model_Deployment.ipynb
    ├── SF_9_Model_Validation.ipynb
    ├── metadata/                       ← YAML config for each stage
    └── python_nodes_library/           ← @sf.node functions
        ├── data_acquisition/
        │   └── load_mon.py             ← parse .mon files + merge monitor stations
        ├── data_cleansing/
        │   └── clean.py               ← drop NaN, report statistics
        ├── data_partition/
        │   └── partition.py           ← 70/10/20 split
        ├── feature_selection/
        │   └── preprocess.py          ← StandardScaler (no OHE)
        ├── model_training/
        │   ├── learn.py               ← train() — MLP + GradientBoosting
        │   └── estimators.py          ← MultiOutputGradientBoosting wrapper
        ├── model_deployment/
        │   └── model.py               ← package sklearn Pipeline
        └── model_validation/
            ├── split_val.py           ← VTP split quality check
            ├── prediction.py          ← predict()
            ├── score.py               ← calculate_metrics(), distribution_tests()
            ├── validation.py          ← validate() — requirement check
            ├── visualize.py           ← scatter + ratio plots
            ├── export_validation_csvs.py
            ├── validation_script.py   ← runs validation_template → HTML
            └── validation_template.ipynb
```

---

## Production Run (real CFD data)

1. Replace `datasets/TrainData/` with actual `.mon` files from the CFD solver
2. Verify `monitor_stations` config in `SF_2_Data_Acquisition_Generation.yaml` matches the real file column layout
3. Adjust `input_data` path in `pipeline_config.yaml` if needed
4. Delete `pipeline/data/` to force a fresh run
5. Run via Elyra or `python UCLoads/pipeline/run_pipeline.py`
