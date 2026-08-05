# UCLoads — Structural Load Surrogate from CFD Monitor Files

## Overview

UCLoads trains a surrogate model that predicts 8 aerodynamic structural loads at two monitor stations from three flight condition inputs (Mach, altitude, angle of attack). Training data comes from Tecplot `.mon` files produced by a CFD solver. Five algorithms are compared — PyTorchNN achieves the best R² on 7 of 8 outputs with no overfitting. A full description of the validation methodology and model comparison is in [LOADS.md](LOADS.md).

---

## Objective

Build a surrogate that approximates:

```
f(Mach, Altitude, ALPHA) → (Fz_WINGROOT, Mx_WINGROOT, My_WINGROOT, Fy_FUS, Fz_FUS, Mx_FUS, My_FUS, Mz_FUS)
```

Achieving Q90 relative error < 10 % on all 8 outputs, enabling rapid structural load screening across the flight envelope without re-running CFD.

---

## Dataset

| Property | Value |
|---|---|
| Source | Tecplot `.mon` files from CFD solver |
| Monitor stations | MS_WINGROOT_RHS (wing root), MS_FUS_FTJ (fuselage frame) |
| Current dataset | 2 000 synthetic rows calibrated to real statistics |
| Production dataset | Replace `datasets/TrainData/` with real CFD monitor outputs |
| Split | 70 % train / 10 % val / 20 % test (`random_state=42`) |

### Inputs (3 features)

| Variable | Unit | Range (synthetic) | Description |
|---|---|---|---|
| `Mach[-]` | — | 0.40 – 1.40 | Flight Mach number |
| `Altitude[ft]` | ft | 0 – 50 000 | Flight altitude (ISA) |
| `ALPHA[deg]` | deg | −7.5 – 32.5 | Angle of attack |

### Outputs (8 targets)

| Variable | Station | Description |
|---|---|---|
| `Fz_WINGROOT` | MS_WINGROOT_RHS | Vertical force at wing root (N) |
| `Mx_WINGROOT` | MS_WINGROOT_RHS | Rolling moment at wing root (Nm) |
| `My_WINGROOT` | MS_WINGROOT_RHS | Pitching moment at wing root (Nm) |
| `Fy_FUS` | MS_FUS_FTJ | Lateral force at fuselage frame (N) |
| `Fz_FUS` | MS_FUS_FTJ | Vertical force at fuselage frame (N) |
| `Mx_FUS` | MS_FUS_FTJ | Rolling moment at fuselage frame (Nm) |
| `My_FUS` | MS_FUS_FTJ | Pitching moment at fuselage frame (Nm) |
| `Mz_FUS` | MS_FUS_FTJ | Yaw moment at fuselage frame (Nm) |

---

## Pipeline Architecture

| Stage | Notebook | Description |
|---|---|---|
| SF_1 | `SF_1_Requirements.ipynb` | Define accuracy targets (Q90 < 0.10 per output) |
| SF_2 | `SF_2_Data_Acquisition.ipynb` | Parse `.mon` files; merge WINGROOT + FUS by row index |
| SF_3 | `SF_3_Data_Cleaning.ipynb` | Drop NaN rows (all-numerical data) |
| SF_4 | `SF_4_Data_Partitioning.ipynb` | 70 / 10 / 20 % train / val / test split |
| SF_5 | `SF_5_Feature_Selection.ipynb` | StandardScale 3 numerical inputs → 3 scaled features |
| SF_6 | `SF_6_Model_Selection.ipynb` | Define MLP, GradientBoosting, RandomForest, XGBoost, PyTorchNN |
| SF_7 | `SF_7_Model_Training.ipynb` | Train all 5 models; log loss curves to MLflow |
| SF_8 | `SF_8_Model_Deployment.ipynb` | Package scaler + model into sklearn Pipeline (.pkl) |
| SF_9 | `SF_9_Model_Validation.ipynb` | Metrics, distribution tests, scatter/ratio plots, HTML report |

---

## Models

### Recommended: PyTorchNN (Residual Deep Network)

Best R² on 7 of 8 outputs; perfect KS test (8/8 outputs — no overfitting). Requires Y-standardisation before training.

| Hyperparameter | Value |
|---|---|
| Architecture | 3 → 128 → 4× ResBlock(128) → 8 |
| Block | Linear → LayerNorm → GELU → Linear → LayerNorm + skip + GELU |
| Optimiser | Adam (lr=0.001, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Batch size | 64 |
| Max epochs | 500 |
| Early stopping | Patience 30 |
| Y-normalisation | StandardScaler per output (fit inside `fit()`) |

### Comparison Models

| Model | n_estimators / Architecture | Notes |
|---|---|---|
| GradientBoosting | 100 trees, max_depth=4, lr=0.1 | Best R² baseline; overfits on synthetic (KS: 0/8) |
| RandomForest | 200 trees, native multi-output | KS: 5/8 ✓; slowest inference |
| XGBoost | 300 trees, max_depth=5, lr=0.05 | Weaker than GB on most outputs; `nthread=1` on macOS |
| MLP | 3 → 64 → 32 → 8, ReLU, Adam | R² fails on My outputs; perfect KS (8/8) |

---

## How to Run

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
MLFLOW_ALLOW_FILE_STORE=true python UCLoads/pipeline/run_pipeline.py
```

**Option C — notebook by notebook:**
Open each `SF_N_*.ipynb` in order using the `UCLoads (Pipeline)` kernel.

---

## Results (synthetic data, 2 000 rows)

> Q90 targets are not met on synthetic data — expected, because the synthetic generator includes 10 % decorrelated noise per output. On real deterministic CFD data, Q90 < 0.10 is expected to be achievable by PyTorchNN and GradientBoosting.

| Output | PyTorchNN R² | GB R² | MLP R² | PyTorchNN Q90 |
|---|---|---|---|---|
| Fz_WINGROOT | **0.990** | 0.983 | 0.912 | 0.271 |
| Mx_WINGROOT | **0.990** | 0.982 | 0.918 | 0.258 |
| My_WINGROOT | **0.991** | 0.988 | 0.540 | 0.193 |
| Fy_FUS | **0.991** | 0.979 | 0.903 | 0.202 |
| Fz_FUS | **0.998** | 0.991 | 0.921 | 0.142 |
| Mx_FUS | **0.998** | 0.991 | 0.922 | 0.134 |
| My_FUS | 0.997 | **0.995** | 0.547 | 0.101 |
| Mz_FUS | **0.987** | 0.975 | 0.883 | 0.252 |

> Full model comparison and extended validation analysis is in [LOADS.md](LOADS.md).

---

## Repository Structure

```
UCLoads/
├── README.md                           ← this file (summary)
├── LOADS.md                            ← detailed validation report and model comparison
├── datasets/
│   └── TrainData/
│       ├── MS_WINGROOT_RHS.mon         ← wing root loads (synthetic, 2 000 rows)
│       └── MS_FUS_FTJ.mon              ← fuselage loads (synthetic, 2 000 rows)
└── pipeline/
    ├── pipeline_config.yaml
    ├── UCLoads.pipeline                ← Elyra visual pipeline graph
    ├── run_pipeline.py                 ← standalone runner
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
        │   └── load_mon.py             ← parse .mon files + merge monitor stations
        ├── data_cleansing/
        ├── feature_selection/
        ├── model_training/
        │   └── estimators.py           ← MultiOutputGradientBoosting wrapper
        ├── model_deployment/
        └── model_validation/
            ├── validation_script.py    ← runs validation_template → HTML report
            └── validation_template.ipynb
```
