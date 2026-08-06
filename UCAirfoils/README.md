# UCAirfoils — Aerodynamic Surrogate for Airfoil Design Space Exploration

## Overview

UCAirfoils trains a surrogate model that maps 18 Kulfan CST airfoil shape coefficients plus aerodynamic flow conditions to lift, drag, and pitching moment coefficients. Training data is generated with **NeuralFoil** (a fast panel-method surrogate for XFOIL), enabling dense sampling of the design space without running expensive CFD or experimental campaigns.

---

## Objective

Build an MLP surrogate that approximates:

```
f(α, Re, Kulfan_1…18) → (Cl, Cd, Cm)
```

Achieving Q90 relative error < 10 % on all three outputs, enabling rapid aerodynamic screening of arbitrary airfoil shapes within the design space.

---

## Dataset

| Property | Value |
|---|---|
| Source | NeuralFoil (XFOIL panel-method surrogate) |
| File | `data/airfoils_data.csv` |
| Rows | 7 987 (filtered: `analysis_confidence ≥ 0.3`) |
| Split | 70 % train / 10 % val / 20 % test (`random_state=42`) |
| Baseline geometry | NACA 0012 (upper_weights ≈ [0.17]×8, lower_weights ≈ [−0.17]×8) |

### Inputs (20 features)

| Variable | Type | Range | Description |
|---|---|---|---|
| `alpha` | Continuous | −11.5° to +11.5° | Angle of attack |
| `Re` | Continuous | 1 × 10⁶ to 20 × 10⁶ | Reynolds number |
| `Kulfan_1`…`Kulfan_8` | Continuous | [−0.2, +0.2] | Upper surface CST weights |
| `Kulfan_9`…`Kulfan_16` | Continuous | [−0.2, +0.2] | Lower surface CST weights |
| `Kulfan_17` | Continuous | [−0.2, +0.2] | Leading edge weight |
| `Kulfan_18` | Continuous | [0, 0.02] | Trailing edge thickness |

### Outputs (3 targets)

| Variable | Description |
|---|---|
| `Cl` | Lift coefficient |
| `Cd` | Drag coefficient |
| `Cm` | Pitching moment coefficient |

---

## Pipeline Architecture

| Stage | Notebook | Description |
|---|---|---|
| SF_1 | `SF_1_Requirements.ipynb` | Define accuracy targets (Q90 < 0.10 per output) |
| SF_2 | `SF_2_Data_Acquisition.ipynb` | Load `airfoils_data.csv`; log dataset statistics |
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

All 20 inputs are continuous and the aerodynamic outputs vary smoothly over the design space — ideal conditions for a deep MLP. GradientBoosting is not recommended here: no discrete inputs and a smooth manifold favour neural networks.

| Hyperparameter | Value |
|---|---|
| Architecture | 20 → 128 → 64 → 32 → 3 |
| Activation | ReLU |
| Solver | Adam |
| L2 regularisation (α) | 0.001 |
| Batch size | 256 |
| Learning rate | Adaptive (init 0.001) |
| Max iterations | 500 |
| Early stopping | Yes — patience 20 epochs, tol 1 × 10⁻⁶ |

---

## How to Run

**Option A — Elyra visual editor (recommended):**
```bash
~/Desktop/Pipelines/start_jupyter.sh
# Open UCAirfoils.pipeline → Run Pipeline (local)
```

**Option B — standalone script:**
```bash
cd ~/Desktop/Pipelines
source .venv/bin/activate
MLFLOW_ALLOW_FILE_STORE=true python UCAirfoils/pipeline/run_pipeline.py
```

**Option C — notebook by notebook:**
Open each `SF_N_*.ipynb` in order.

---

## Results

> Results on the full 7 987-row dataset generated with NeuralFoil.

| Output | Requirement | Status |
|---|---|---|
| Cl | Q90 < 0.10 | Run SF_9 to evaluate |
| Cd | Q90 < 0.10 | Run SF_9 to evaluate |
| Cm | Q90 < 0.10 | Run SF_9 to evaluate |

The surrogate is validated through the full SF_9 pipeline: split quality (VTP method), R², MAE, Q90 per output, KS distribution tests, scatter/ratio plots, and an HTML report generated via `validationlib`.

---

## Repository Structure

```
UCAirfoils/
├── README.md
├── data/
│   └── airfoils_data.csv               ← 7 987 rows generated with NeuralFoil
├── images/
│   └── *.png                           ← dispersion matrix, Cl/Cd/Cm surfaces
└── pipeline/
    ├── pipeline_config.yaml
    ├── UCAirfoils.pipeline             ← Elyra visual pipeline graph
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
        ├── data_cleansing/
        ├── feature_selection/
        ├── model_training/
        ├── model_deployment/
        └── model_validation/
```
