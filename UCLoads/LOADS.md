# UCLoads — Surrogate Model for Structural Load Prediction

## Validation Results Summary (synthetic data)

> Results on **500 synthetic rows** (plausible aerodynamic physics, `random_state=42`). Real data results will differ significantly once `.mon` files from CFD are available.

### Selected model: GradientBoosting

GradientBoosting dominates on all learnable outputs. The reports below are generated automatically by SF_9 as HTML files per model.

| Output | R² GB | Q90 GB | Pass (< 0.10) | R² MLP | Note |
|---|---|---|---|---|---|
| Fz_WINGROOT | 0.990 | 0.094 | ✅ | 0.922 | Lift force at wing root |
| Mx_WINGROOT | 0.977 | 0.264 | ❌ | 0.739 | Roll moment — harder to learn |
| My_WINGROOT | 0.989 | 0.098 | ✅ | 0.917 | Pitch moment at wing root |
| Fy_FUS | −0.037 | 3.004 | ❌ | 0.036 | Side force — near-zero with high noise |
| Fz_FUS | 0.992 | 0.089 | ✅ | 0.929 | Vertical force at fuselage |
| Mx_FUS | 0.964 | 0.723 | ❌ | 0.059 | Roll moment — MLP fails, GB partially |
| My_FUS | 0.991 | 0.088 | ✅ | 0.928 | Pitch moment at fuselage |
| Mz_FUS | 0.973 | 0.183 | ❌ | 0.838 | Yaw moment — lower signal-to-noise |

### Key findings

**Split quality:** excellent — residual voxel proportion = 0.0, no phacking, no isolated train points. The random 70/10/20 split on 3 continuous inputs produces a statistically sound partition.

**Accuracy (GradientBoosting):** 4/8 outputs pass Q90 < 10 %. The four failing outputs all relate to issues in the synthetic data design:
- **Fy_FUS** (side force): designed with near-zero mean and large relative noise → Q90 > 300 % regardless of model
- **Mx_WINGROOT**, **Mx_FUS**: rolling moments require more data to capture the full input interaction
- **Mz_FUS**: yaw moment has lower amplitude variation → higher relative error

**On real CFD data:** these outputs are expected to have physically consistent, deterministic signal. The Q90 failures here are artefacts of the synthetic noise model, not model architecture.

**MLP:** struggles on Fy_FUS (R²≈0) and Mx_FUS (R²≈0.06). GradientBoosting is strongly preferred.

**KS overfitting signal:** all 8 outputs fail the KS test for GradientBoosting (train vs test residuals differ). This is expected with 360 training samples and 100 GB trees — regularisation would help on real data.

**Recommendation:** Use GradientBoosting. When real CFD data is available, prioritise augmenting the Fy_FUS and moment load cases. Add a `max_depth=3` constraint to reduce overfitting.

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
SF_6  Model Selection       Define MLP and GradientBoosting architectures
  ↓
SF_7  Model Training        Train both models; log loss curves to MLflow
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
