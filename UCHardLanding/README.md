# UCHardLanding — Structural Load Surrogate for Hard Landing Analysis

## Overview

UCHardLanding trains a surrogate model that predicts two structural outputs (O1, O2) from seven landing configuration inputs (I1–I7). The dataset consists of ~89k pre-run structural simulation results covering a wide design space of discrete and continuous landing parameters. GradientBoosting is the recommended model due to the predominantly discrete nature of the inputs.

---

## Objective

Build a surrogate that approximates:

```
f(I1, I2, I3, I4, I5, I6, I7) → (O1, O2)
```

Achieving Q90 relative error < 10 % on both outputs, enabling real-time structural screening across landing configurations without re-running the full simulation.

---

## Dataset

| Property | Value |
|---|---|
| Source | Structural simulation outputs (CSV export) |
| File | `data/datos.csv` (semicolon-separated) |
| Rows | 89 357 |
| Split | 70 % train / 10 % val / 20 % test (`random_state=42`) |
| Preprocessing | Values pre-normalised to [0, 1] in the source file |

### Inputs (7 features)

| Variable | Type | Levels | Description |
|---|---|---|---|
| `I1` | Discrete | 8 | Landing configuration parameter 1 |
| `I2` | Discrete | 14 | Landing configuration parameter 2 |
| `I3` | Discrete | 11 | Landing configuration parameter 3 |
| `I4` | Discrete | 4 | Landing configuration parameter 4 |
| `I5` | Discrete | 3 | Landing configuration parameter 5 |
| `I6` | Discrete | 2 | Landing configuration parameter 6 |
| `I7` | Continuous | — | Continuous landing parameter (dominant: r ≈ 0.85–0.92 with outputs) |

### Outputs (2 targets)

| Variable | Description |
|---|---|
| `O1` | Structural load output 1 |
| `O2` | Structural load output 2 |

---

## Pipeline Architecture

| Stage | Notebook | Description |
|---|---|---|
| SF_1 | `SF_1_Requirements.ipynb` | Define accuracy targets (Q90 < 0.10 for O1 and O2) |
| SF_2 | `SF_2_Data_Acquisition.ipynb` | Load `datos.csv` (sep=`;`); log dataset statistics |
| SF_3 | `SF_3_Data_Cleaning.ipynb` | Verify no missing values (data is pre-clean) |
| SF_4 | `SF_4_Data_Partitioning.ipynb` | 70 / 10 / 20 % train / val / test split |
| SF_5 | `SF_5_Feature_Selection.ipynb` | Fit `normalizer_transformer` (MinMax) — inputs already in [0,1] |
| SF_6 | `SF_6_Model_Selection.ipynb` | Define GradientBoosting (recommended) and MLP architectures |
| SF_7 | `SF_7_Model_Training.ipynb` | Train both models; log loss/score curves to MLflow |
| SF_8 | `SF_8_Model_Deployment.ipynb` | Package scaler + model into sklearn Pipeline (.pkl) |
| SF_9 | `SF_9_Model_Validation.ipynb` | Metrics, distribution tests, scatter/ratio plots, HTML report |

---

## Models

### Recommended: GradientBoosting

Six of seven inputs are discrete. Decision trees split naturally at each discrete level boundary, making GradientBoosting the preferred algorithm. I7 (continuous) is the dominant predictor (Pearson r ≈ 0.85–0.92 with both outputs).

| Hyperparameter | Value |
|---|---|
| Estimators per output | 300 |
| Max depth | 5 |
| Learning rate | 0.05 |
| Subsample | 0.8 |
| Min samples leaf | 10 |
| Wrapper | `MultiOutputRegressor` (one model per output) |

### Comparison: MLP Regressor

Kept for comparison. MLP treats all inputs as continuous and must learn step-function behaviour from gradient descent — less efficient than tree splits for discrete inputs.

| Hyperparameter | Value |
|---|---|
| Architecture | 7 → 64 → 32 → 2 |
| Activation | ReLU |
| Solver | Adam |
| L2 regularisation (α) | 0.01 |
| Batch size | 128 |
| Max iterations | 500 |
| Early stopping | Yes — patience 20 epochs |

---

## How to Run

**Option A — Elyra visual editor (recommended):**
```bash
~/Desktop/Pipelines/start_jupyter.sh
# Open UCHardLanding.pipeline → Run Pipeline (local)
```

**Option B — standalone script:**
```bash
cd ~/Desktop/Pipelines
source .venv/bin/activate
MLFLOW_ALLOW_FILE_STORE=true python UCHardLanding/pipeline/run_pipeline.py
```

**Option C — notebook by notebook:**
Open each `SF_N_*.ipynb` in order.

---

## Results

> Results on the full 89 357-row dataset.

| Output | Requirement | Status |
|---|---|---|
| O1 | Q90 < 0.10 | Run SF_9 to evaluate |
| O2 | Q90 < 0.10 | Run SF_9 to evaluate |

With 89k rows and predominantly discrete inputs, GradientBoosting (n_estimators=300, max_depth=5) is expected to comfortably meet the Q90 < 0.10 requirement on both outputs. MLP results will serve as baseline comparison.

---

## Repository Structure

```
UCHardLanding/
├── README.md
├── data/
│   └── datos.csv                       ← 89 357 rows, semicolon-separated, pre-normalised [0,1]
└── pipeline/
    ├── pipeline_config.yaml
    ├── UCHardLanding.pipeline          ← Elyra visual pipeline graph
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
        │   └── estimators.py           ← MultiOutputGradientBoosting wrapper
        ├── model_deployment/
        └── model_validation/
```
