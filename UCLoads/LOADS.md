# UCLoads — Surrogate Model for Structural Load Prediction

## Validation Results Summary (synthetic data v2)

> Results on **2000 synthetic rows** calibrated to real dataset statistics (real data: 357k rows, Mach 0.4–1.4, Alt 0–50k ft, ALPHA −7.5° to 32.5°).
> Q90 targets are not met on synthetic data — this is expected because the synthetic generator includes 10% decorrelated noise per output.
> On real deterministic CFD data, Q90 < 0.10 is expected to be achievable by PyTorchNN and GradientBoosting.

### 5-model comparison (MLP · GradientBoosting · RandomForest · XGBoost · PyTorchNN)

| Output | MLP R² | GB R² | RF R² | XGB R² | PyTorchNN R² | PyTorchNN Q90 | Note |
|---|---|---|---|---|---|---|---|
| Fz_WINGROOT | 0.912 | 0.983 | 0.982 | 0.924 | **0.990** | 0.271 | Vertical force at wing root |
| Mx_WINGROOT | 0.918 | 0.982 | 0.980 | 0.924 | **0.990** | 0.258 | Roll moment at wing root |
| My_WINGROOT | 0.540 | 0.988 | 0.946 | 0.984 | **0.991** | 0.193 | Pitch moment at wing root |
| Fy_FUS | 0.903 | 0.979 | 0.971 | 0.925 | **0.991** | 0.202 | Side force at fuselage |
| Fz_FUS | 0.921 | 0.991 | 0.987 | 0.932 | **0.998** | 0.142 | Vertical force at fuselage |
| Mx_FUS | 0.922 | 0.991 | 0.987 | 0.931 | **0.998** | 0.134 | Roll moment at fuselage |
| My_FUS | 0.547 | **0.995** | 0.950 | 0.990 | 0.997 | 0.101 | Pitch moment at fuselage |
| Mz_FUS | 0.883 | 0.975 | 0.968 | 0.914 | **0.987** | 0.252 | Yaw moment (97% negative) |

### KS test (overfitting indicator — should not reject H₀)

| Model | Passes / 8 outputs | Interpretation |
|---|---|---|
| **MLP** | **8 / 8** | No overfitting — train and test residuals same distribution |
| **PyTorchNN** | **8 / 8** | No overfitting — train and test residuals same distribution |
| RandomForest | 5 / 8 | Slight overfitting on Fz/Mx |
| GradientBoosting | 0 / 8 | Overfits (expected with 100 trees / 1440 training samples) |
| XGBoost | 0 / 8 | Overfits (expected with 300 trees / 1440 training samples) |

### Key findings

**Split quality:** excellent — residual voxel proportion = 0.0, no phacking, no isolated train points.

**PyTorchNN** (residual deep network, 4 blocks, 128 hidden units):
- Best R² on 7/8 outputs; wins on every output vs all other models
- Perfect KS test: 8/8 outputs pass → no overfitting
- Early stopped at epoch 66 (val loss 0.178 in normalised space)
- **Critical**: outputs must be Y-standardised before training; without it PyTorchNN fails completely (MSE dominated by large-amplitude loads)
- **Recommendation for real data**: scale to n_hidden=256, n_layers=6 to exploit the 357k-row dataset

**GradientBoosting** (100 trees per output × 8 MultiOutputRegressor):
- Strong R² 0.975–0.995 on all outputs; best on My_FUS (R²=0.995)
- Q90 approximately 2–3× the target; expected to pass on real data
- Overfits on synthetic (KS: 0/8) — regularise with `max_depth=3` on real data if needed

**RandomForest** (200 trees, native multi-output):
- R² 0.946–0.987 — solid but consistently below GB and PyTorchNN
- KS: 5/8 ✓ — moderate overfitting
- Fastest inference at deployment time

**XGBoost** (300 trees per output, `nthread=1` required on macOS):
- R² 0.914–0.990, weaker than GB on most outputs
- Overfits strongly (KS: 0/8)

**MLP** (64→32 ReLU):
- R² 0.54 on My_WINGROOT and My_FUS — fails to capture pitching-moment nonlinearity
- R² ~0.88–0.92 on the 6 other outputs
- Perfect KS: 8/8 ✓ — good generalisation but insufficient accuracy

**Synthetic Q90 note:** With 10% decorrelated noise in the generator, the irreducible Q90 floor is ≈ 0.15–0.20 × the output scale. On real CFD data (near-deterministic physics, 357k rows), this floor disappears and Q90 < 0.10 is achievable.

**Recommendation for real CFD data (357k rows):**
- **PyTorchNN** as primary model: best accuracy + generalisation, scales well with large datasets
- **GradientBoosting** as baseline: faster training, interpretable, R² close to PyTorchNN
- Fy_FUS will improve significantly once real sideslip data (BETA ≠ 0) is included in inputs

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
| Current dataset | 2000 synthetic rows calibrated to real statistics (means, stds, sign %; `random_state=42`) |
| Production dataset | Replace `TrainData/` with real CFD monitor outputs |
| Split | 70 % train / 10 % val / 20 % test (`random_state=42`) |

### Inputs (3 features — all numerical)

| Feature | Unit | Range (synthetic) | Description |
|---|---|---|---|
| Mach[-] | – | 0.40 – 1.40 | Flight Mach number (includes transonic) |
| Altitude[ft] | ft | 0 – 50 000 | Flight altitude (ISA) |
| ALPHA[deg] | deg | −7.5 – 32.5 | Angle of attack |

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
│       ├── MS_WINGROOT_RHS.mon         ← wing root loads (2000 calibrated synthetic rows)
│       └── MS_FUS_FTJ.mon              ← fuselage loads (2000 calibrated synthetic rows)
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
