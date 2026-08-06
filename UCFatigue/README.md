# UCFatigue — Fatigue Load Surrogate for Aircraft Structural Analysis

## Overview

UCFatigue trains a surrogate model that predicts 7 fatigue load outputs for a structural element from 8 flight-condition inputs. The dataset is produced by finite-element (FEM) simulations. GradientBoosting is the recommended model, achieving R² ≥ 0.929 on 5 of 7 outputs on the development subset. A full description of the validation methodology and extended analysis is in [FATIGUE.md](FATIGUE.md).

---

## Objective

Build a surrogate that approximates:

```
f(FLAP, Altitude, TAS, Mass, q, gamma, Type_segment, Xcg) → (1g, Vert. maneuver, Vert. gust, Turn, Frontal gust, n0, Giro)
```

Achieving Q90 relative error < 10 % on all 7 outputs, enabling real-time fatigue load estimation across the flight envelope without running a full FEM simulation.

---

## Dataset

| Property | Value |
|---|---|
| Source | FEM simulation outputs (Excel export) |
| File | `datasets/Subset_Surrogate_Factory.xlsx` |
| Element | SSE 2110017 (single structural element, development run) |
| Rows | ~870 (development subset) |
| Split | 70 % train / 10 % val / 20 % test (`random_state=42`) |

### Inputs (8 features → 19 after encoding)

| Variable | Type | Description |
|---|---|---|
| `FLAP` | Categorical (4 levels) | Flap position (0 / 10 / 15 / 23°) |
| `Altitude` | Continuous | Flight altitude (ft) |
| `TAS` | Continuous | True airspeed (kt) |
| `Mass` | Continuous | Aircraft mass (kg) |
| `q` | Continuous | Dynamic pressure |
| `gamma` | Continuous | Flight path angle |
| `Type_segment` | Categorical (11 levels) | Flight segment type (FLT-1…FLT-11) |
| `Xcg(%CMA)` | Continuous | Centre-of-gravity position (% MAC) |

### Outputs (7 targets)

| Variable | Description |
|---|---|
| `1g` | Level flight fatigue load |
| `Vertical maneuver` | Vertical maneuver load |
| `Vertical gust` | Vertical gust load |
| `Turn` | Turn load |
| `Frontal gust` | Frontal gust load |
| `n0` | Zero-g load |
| `Giro` | Gyroscopic load |

---

## Pipeline Architecture

| Stage | Notebook | Description |
|---|---|---|
| SF_1 | `SF_1_Requirements.ipynb` | Define accuracy targets (Q90 < 0.10 per output) |
| SF_2 | `SF_2_Data_Acquisition.ipynb` | Read Excel, filter by SSE element, parse outputs |
| SF_3 | `SF_3_Data_Cleaning.ipynb` | Fill missing values (Frontal gust → 0, Xcg → median); filter invalid Type_segment |
| SF_4 | `SF_4_Data_Partitioning.ipynb` | 70 / 10 / 20 % train / val / test split |
| SF_5 | `SF_5_Feature_Selection.ipynb` | OneHotEncode FLAP + Type_segment; StandardScale numerical → 19 features |
| SF_6 | `SF_6_Model_Selection.ipynb` | Define GradientBoosting (recommended) and MLP architectures |
| SF_7 | `SF_7_Model_Training.ipynb` | Train both models; log loss/score curves to MLflow |
| SF_8 | `SF_8_Model_Deployment.ipynb` | Package scaler + model into sklearn Pipeline (.pkl) |
| SF_9 | `SF_9_Model_Validation.ipynb` | Metrics, distribution tests, plots, HTML report (validationlib) |

---

## Models

### Recommended: GradientBoosting

One independent `GradientBoostingRegressor` per output, wrapped in `MultiOutputRegressor`.

| Hyperparameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 4 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| min_samples_leaf | 5 |

### Comparison: MLP Regressor

| Hyperparameter | Value |
|---|---|
| Architecture | 19 → 64 → 32 → 7 |
| Activation | ReLU |
| Solver | Adam |
| L2 regularisation (α) | 0.01 |
| Batch size | 32 |
| Max iterations | 500 |
| Early stopping | Yes — patience 20 epochs |

---

## How to Run

**Option A — Elyra visual editor (recommended):**
```bash
~/Desktop/Pipelines/start_jupyter.sh
# Open UCFatigue.pipeline → Run Pipeline (local)
```

**Option B — standalone script:**
```bash
cd ~/Desktop/Pipelines
source .venv/bin/activate
python UCFatigue/pipeline/run_pipeline.py
```

**Option C — notebook by notebook:**
Open each `SF_N_*.ipynb` in order.

---

## Results (development subset, ~870 rows)

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

**Key findings:**
- GradientBoosting achieves R² ≥ 0.929 on 5 of 7 outputs.
- **Frontal gust** and **Giro** fail Q90 < 0.10 — both are low-amplitude loads with sparse representation in the development subset. More simulation data in those regimes is needed.
- MLP shows negative R² on `n0` (−0.73) and `Giro` (−0.10) — unsuitable for production.
- Results on the **full dataset** are expected to improve significantly.

> Full extended validation analysis (split quality, bias, uncertainty models, KS tests) is in [FATIGUE.md](FATIGUE.md).

---

## Repository Structure

```
UCFatigue/
├── README.md                           ← this file (summary)
├── FATIGUE.md                          ← detailed validation report and analysis
├── datasets/
│   └── Subset_Surrogate_Factory.xlsx   ← development subset (~870 rows)
└── pipeline/
    ├── pipeline_config.yaml
    ├── UCFatigue.pipeline              ← Elyra visual pipeline graph
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
            ├── validation_script.py    ← runs validation_template → HTML report
            └── validation_template.ipynb
```
