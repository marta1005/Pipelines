# UCFatigue — Surrogate Model for Fatigue Load Prediction

## Overview

UCFatigue is a machine-learning surrogate that replaces expensive finite-element (FEM) simulations for predicting fatigue loads on aircraft structural elements. Given a set of flight parameters, the model predicts 7 fatigue-load outputs in milliseconds instead of the hours required by a full structural simulation.

The pipeline is built on **Surrogate Factory v2.2**, an Airbus Inner Source MLOps framework that structures the development process into 9 reproducible stages managed through Jupyter notebooks and an Elyra visual pipeline editor.

---

## Problem Statement

Fatigue analysis requires running a structural simulation for every combination of:
- Flight condition (altitude, speed, mass, load factor…)
- Aircraft configuration (flap setting, CG position, flight segment type)

A surrogate model trained on a representative sample of simulations can approximate the full mapping at negligible compute cost, enabling rapid design-space exploration and load-case screening.

---

## Dataset

| Property | Value |
|---|---|
| Source | FEM simulation outputs (Excel export) |
| Element | SSE 2110017 (single structural element, current run) |
| Development subset | ~870 rows |
| Full dataset | Larger — use full file on company infrastructure |
| Split | 70 % train / 10 % val / 20 % test (`random_state=42`) |

### Inputs (8 features)

| Feature | Type | Description |
|---|---|---|
| FLAP | Categorical | Flap position (0 / 10 / 15 / 23°) |
| Altitude | Numerical | Flight altitude (ft) |
| TAS | Numerical | True airspeed (kt) |
| Mass | Numerical | Aircraft mass (kg) |
| q | Numerical | Dynamic pressure |
| gamma | Numerical | Flight path angle |
| Type_segment | Categorical | Flight segment type (FLT-1 … FLT-11) |
| Xcg(%CMA) | Numerical | Centre-of-gravity position (% MAC) |

### Outputs (7 fatigue loads)

| Output | Description |
|---|---|
| 1g | Level flight fatigue load |
| Vertical maneuver | Vertical maneuver load |
| Vertical gust | Vertical gust load |
| Turn | Turn load |
| Frontal gust | Frontal gust load |
| n0 | Zero-g load |
| Giro | Gyroscopic load |

---

## Pipeline Architecture

The pipeline follows the 9-stage Surrogate Factory structure. Each stage is a Jupyter notebook (`SF_N_*.ipynb`) that reads and writes metadata through a shared `Workflow` object.

```
SF_1  Requirements          Define accuracy targets
  ↓
SF_2  Data Acquisition      Read Excel, filter by SSE element, parse outputs
  ↓
SF_3  Data Cleansing        Fill missing values (Frontal gust → 0, Xcg → median)
                            Filter invalid Type_segment values
  ↓
SF_4  Data Partitioning     70 / 10 / 20 % train / val / test split
  ↓
SF_5  Feature Selection     OneHotEncode FLAP + Type_segment
                            StandardScale numerical features
                            → 19 features after encoding
  ↓
SF_6  Model Selection       Define MLP and GradientBoosting architectures
  ↓
SF_7  Model Training        Train both models, save .modl artifacts
  ↓
SF_8  Model Deployment      Save sklearn Pipeline (.modl) with preprocessor + model
  ↓
SF_9  Model Validation      Split quality, predictions, metrics, distribution tests,
                            requirement check, scatter/ratio plots
```

### Running the pipeline

**Option A — Elyra visual editor (recommended):**
```bash
~/Desktop/Pipelines/start_jupyter.sh
# Open UCFatigue.pipeline → Run Pipeline (local)
# Kernel: UCFatigue (Pipeline)
```

**Option B — standalone script:**
```bash
cd ~/Desktop/Pipelines
source .venv/bin/activate
python UCFatigue/pipeline/run_pipeline.py
```

**Option C — notebook-by-notebook:**  
Open each `SF_N_*.ipynb` in order using the `UCFatigue (Pipeline)` kernel.

---

## Models

### MLP Regressor

Single multi-output network trained jointly on all 7 outputs.

| Hyperparameter | Value |
|---|---|
| Architecture | 19 → 64 → 32 → 7 |
| Activation | ReLU |
| Solver | Adam |
| L2 regularisation (α) | 0.01 |
| Batch size | 32 |
| Learning rate | Adaptive (init 0.001) |
| Max iterations | 500 |
| Early stopping | Yes — patience 20 epochs, tol 1e-5 |

### Gradient Boosting

One independent `GradientBoostingRegressor` per output (7 models), wrapped in `MultiOutputRegressor`.

| Hyperparameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 4 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| min_samples_leaf | 5 |

---

## Accuracy Requirements

All 7 outputs must satisfy **quantile 90 of relative error < 0.1** (i.e. 90 % of predictions within ±10 % of the true value).

---

## Results (development subset)

> **Note:** These results are on the development subset (~870 rows). Results on the full dataset will differ.

### GradientBoosting

| Output | R² | Q90 error | Requirement (< 0.10) |
|---|---|---|---|
| 1g | 0.980 | 0.026 | ✓ |
| Vertical maneuver | 0.972 | 0.032 | ✓ |
| Vertical gust | 0.991 | 0.026 | ✓ |
| Turn | 0.929 | 0.035 | ✓ |
| Frontal gust | 0.996 | 0.280 | ✗ |
| n0 | 0.932 | 0.015 | ✓ |
| Giro | 0.950 | 0.222 | ✗ |

### MLP

| Output | R² | Q90 error | Requirement (< 0.10) |
|---|---|---|---|
| 1g | 0.916 | 0.047 | ✓ |
| Vertical maneuver | 0.912 | 0.055 | ✓ |
| Vertical gust | 0.938 | 0.092 | ✓ |
| Turn | 0.754 | 0.078 | ✓ |
| Frontal gust | 0.979 | 11.33 | ✗ |
| n0 | −0.729 | 0.072 | ✓ |
| Giro | −0.104 | 0.901 | ✗ |

### Analysis

- **GradientBoosting dominates** on all outputs. R² between 0.929 and 0.996.
- **Frontal gust** and **Giro** fail the Q90 < 0.10 requirement for both models. These outputs likely have high variance or sparse representation in the current subset.
- **MLP** shows poor R² on `n0` (−0.73) and `Giro` (−0.10), suggesting these outputs require more data or tuning.
- Results on the **full dataset** are expected to improve significantly, particularly for the failing outputs.

---

## Validation Methodology

Stage 9 applies four layers of validation:

| Section | Method | What it checks |
|---|---|---|
| 9.0 | Voxel Tesselation Proximity (validationlib) | Train/test split quality — no phacking, no isolated test points |
| 9.0b | doubleHistogram + PCA scatter (validationlib) | Input/output distribution similarity across train / val / test |
| 9.2 | R², MAE, quantile90 | Prediction accuracy on test set |
| 9.2b | Kolmogorov-Smirnov test (validationlib) | Train vs test residual distributions — detects overfitting |
| 9.2c | doubleHistogram + doublecumulative (validationlib) | Residual distribution visualisation per model |
| 9.3 | Requirement check | Pass/fail table against Q90 < 0.10 target |
| 9.4 | Scatter + ratio plots | y_pred vs y_true, y_pred/y_true per output |

---

## Repository Structure

```
UCFatigue/
├── FATIGUE.md                          ← this file
├── datasets/
│   └── Subset_Surrogate_Factory.xlsx   ← development subset
└── pipeline/
    ├── pipeline_config.yaml            ← paths, catalog, job config
    ├── UCFatigue.pipeline              ← Elyra visual pipeline graph
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
        ├── data_cleansing/
        ├── data_partition/
        ├── feature_selection/
        ├── model_training/
        │   ├── learn.py                ← train() — MLP + GB
        │   └── estimators.py          ← MultiOutputGradientBoosting wrapper
        ├── model_deployment/
        └── model_validation/
            ├── split_val.py            ← split_validation() — VTP method
            ├── prediction.py           ← predict()
            ├── score.py                ← calculate_metrics(), distribution_tests()
            ├── validation.py           ← validate() — requirement check
            └── visualize.py           ← plot() — scatter + ratio plots

validationlib/                          ← Airbus Inner Source validation library
├── misc/
│   ├── split_validation.py             ← voxel_tesselation_proximity_method()
│   └── metrics.py                      ← ratio, residual, abs_residual metrics
├── plots/
│   ├── scatter.py                      ← scatterplot(), binnedScatterplot()
│   ├── hist.py                         ← histogram(), doubleHistogram()
│   └── cumu.py                         ← doublecumulative()
└── tests/
    └── dist.py                         ← dist_similarity() — KS, AD, MW tests

start_jupyter.sh                        ← launch JupyterLab with .venv
.venv/                                  ← isolated Python environment (gitignored)
```

---

## Environment Setup

```bash
# Clone and create the virtual environment
git clone <repo-url>
cd Pipelines
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .                        # installs surrogate_factory
pip install jupyterlab elyra openpyxl scipy ipykernel sympy statsmodels

# Register the Jupyter kernel
python -m ipykernel install --user --name=ucfatigue --display-name="UCFatigue (Pipeline)"

# Launch
./start_jupyter.sh
```

> Use kernel **UCFatigue (Pipeline)** in all notebooks.

---

## Production Run (company infrastructure)

1. Replace `input_data` path in `pipeline_config.yaml` with the full dataset path
2. Set `max_iter: 500` and `n_estimators: 100` in `SF_6_Model_Selection.yaml` (already set)
3. Add `SSE` column to `inputs` list if training across multiple structural elements
4. Delete `pipeline/data/` folder to force a fresh run
5. Run via Elyra or `python UCFatigue/pipeline/run_pipeline.py`
