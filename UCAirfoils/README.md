# UCAirfoils — Aerodynamic Surrogate for Airfoil Design Space Exploration

## Overview

UCAirfoils trains a surrogate that maps an airfoil's Kulfan CST shape parameters and its flow condition to the aerodynamic coefficients and the boundary-layer transition locations. The training set is **generated inside the pipeline**: a Sobol Design of Experiments over the design space, evaluated with **NeuralFoil** (a fast neural surrogate for XFOIL). There is no external dataset to supply.

---

## Objective

```
f(alpha, log10_Re, uW1..8, lW1..8, TE_thickness, LE_weight)
        -> (CL, CD, CM, Top_Xtr, Bot_Xtr)
```

Q90 relative error below 10 % on the force coefficients and below 15 % on the transition locations, so arbitrary sections can be screened without calling XFOIL.

---

## Design space

Declared in `pipeline/metadata/SF_2_Data_Acquisition_Generation.yaml`. 22 entries: 20 sampled, 2 derived, 1 fixed.

### Sampled (20)

| Variable | Family | Bounds | Notes |
|---|---|---|---|
| `alpha` | fluid_mechanics | [−12.0, 12.0] | Angle of attack (deg) |
| `reynolds` | fluid_mechanics | [1e5, 1e8] | **Sampled logarithmically** — see below |
| `uW1`…`uW8` | — | [0.1, 0.36] | Upper surface CST weights |
| `lW1`…`lW8` | — | [−0.34, 0.13] | Lower surface CST weights |
| `TE_thickness` | — | [0.0015, 0.0051] | Trailing edge thickness |
| `LE_weight` | — | [−1.8e−2, 2.3e−1] | Leading edge weight |

### Derived (2)

| Variable | Equation |
|---|---|
| `alpha_rad` | `alpha * 3.1415926545 / 180` |
| `log10_Re` | `log10(reynolds)` |

### Fixed (1)

| Variable | Value |
|---|---|
| `mach` | 0.3 |

### Constraints

```
-0.2 < alpha_rad
alpha_rad < 0.2
```

Which restricts alpha to roughly ±11.46°, inside the ±12° box.

### DoE

| Setting | Value |
|---|---|
| Tool | python library: Sobol |
| Algorithm | Sobol (scrambled, seed 42) |
| Size | 200 000 |
| Margin on bounds | 0 |

---

## Two deliberate departures from the raw specification

**Reynolds is sampled in log space.** The bounds span three decades. A linear Sobol draw puts about 99 % of the points above 1e6 and leaves the low-Re end almost empty. NeuralFoil itself consumes `(log(Re) − 12.5) / 3.5`, so log sampling matches how the physics actually enters. Set `scale: linear` on that entry to sample the interval uniformly instead.

**The model is trained on `log10_Re`, not `reynolds`.** For the same reason: after MinMax scaling, a raw column spanning 1e5–1e8 crushes nearly every point into the bottom of the range.

Also excluded from the model inputs, though present in the dataset:

- `mach` — fixed at 0.3, so it carries no information
- `alpha_rad` — an exact linear function of `alpha`, redundant
- `reynolds` — superseded by `log10_Re`

That leaves **20 model inputs** and **5 outputs**.

---

## Data quality: the confidence filter

NeuralFoil returns an `analysis_confidence` for every evaluation. Random combinations of Kulfan weights produce many implausible sections, and those come back with low confidence **and visibly noisier coefficients**:

| | rows | CD standard deviation |
|---|---|---|
| `analysis_confidence` < 0.5 | ~40 % | 0.0336 |
| `analysis_confidence` ≥ 0.5 | ~60 % | 0.0132 |

Training on the low-confidence points means fitting noise, so SF_3 drops them (`data_cleansing/confidence.py`, threshold 0.5). Of 200 000 generated points, roughly **120 000 survive**.

---

## Pipeline Architecture

| Stage | Notebook | Description |
|---|---|---|
| SF_1 | `SF_1_Requirements.ipynb` | Accuracy targets per output |
| SF_2 | `SF_2_Data_Acquisition.ipynb` | Sobol DoE + NeuralFoil evaluation |
| SF_3 | `SF_3_Data_Cleaning.ipynb` | Confidence filter, then missing values |
| SF_4 | `SF_4_Data_Partitioning.ipynb` | 70 / 10 / 20 % train / val / test |
| SF_5 | `SF_5_Feature_Selection.ipynb` | Fit `normalizer_transformer` (MinMax) |
| SF_6 | `SF_6_Model_Selection.ipynb` | MLP architecture and hyperparameters |
| SF_7 | `SF_7_Model_Training.ipynb` | Train; log to MLflow |
| SF_8 | `SF_8_Model_Deployment.ipynb` | Package scaler + model into a `.pkl` |
| SF_9 | `SF_9_Model_Validation.ipynb` | Metrics, distribution tests, plots, reports |
| SF_10 | `SF_10_Model_Storage.ipynb` | Store artifacts + integration test |

---

## Model

| Hyperparameter | Value |
|---|---|
| Architecture | 20 → 256 → 128 → 64 → 5 |
| Activation | ReLU |
| Solver | Adam |
| L2 (α) | 0.0001 |
| Batch size | 512 |
| Learning rate | Adaptive (init 0.001) |
| Max iterations | 500 |
| Early stopping | Yes — patience 20, tol 1e−6 |

---

## How to Run

No dataset is required; SF_2 generates it.

```bash
~/Desktop/Pipelines/start_jupyter.sh UCAirfoils
# Run SF_1 … SF_10 in order
```

The DoE and NeuralFoil evaluation run at roughly 22 000 points/s, so the full 200 000-point generation takes about 10 seconds. Training dominates the wall clock.

Requires `neuralfoil` (`pip install neuralfoil`).

---

## Results

Full run: 200 000 DoE points, 119 756 after the confidence filter, 83 828 train / 11 976 val / 23 952 test. Training stopped early at iteration 199 (loss 1.3e−4).

| Output | R² | MAE | Q90 | Target | Status |
|---|---|---|---|---|---|
| `CL` | 0.9990 | 0.0127 | 0.0846 | 0.10 | **PASS** |
| `CD` | 0.9416 | 0.00167 | 0.1340 | 0.10 | FAIL |
| `CM` | 0.9800 | 0.00329 | 0.5013 | 0.10 | FAIL |
| `Top_Xtr` | 0.9959 | 0.0119 | 0.1822 | 0.15 | FAIL |
| `Bot_Xtr` | 0.9940 | 0.0139 | 0.2631 | 0.15 | FAIL |

Split quality: residual voxel proportion 0.000, valid test proportion 0.965.

### Reading these numbers

R² is between 0.94 and 0.999 on every output, so the surrogate reproduces the field well. Most of the Q90 failures are the **metric**, not the model: Q90 here is a *relative* error, and three of the five targets pass through or sit on zero.

| Output | Crosses / touches zero | Q90 (all points) | Q90 (excluding the smallest 10 % by magnitude) |
|---|---|---|---|
| `CM` | ranges −0.167 … +0.074; 10.5 % of points have \|CM\| < 0.01 | 0.373 | 0.217 |
| `Top_Xtr` | exactly 0 for 2.0 % of points | 0.233 | 0.105 |
| `Bot_Xtr` | exactly 0 for 2.4 % of points | 0.406 | 0.211 |

Dividing by a target that is essentially zero inflates the error without the prediction being meaningfully wrong — `Top_Xtr` would meet its 0.15 target on the points where the metric is well defined.

`CD` is the genuine weak spot, and the only one not explained this way: excluding small values barely moves it (0.271 → 0.266 on the same recomputation). It is strictly positive and spans roughly a decade, so training on `log(CD)` and inverting at prediction time is the obvious next step.

> The Q90 column above is the pipeline's own figure from `metadata_UCAIRFOILS_1.json`. The comparison table uses a plain \|ŷ−y\|/\|y\| recomputation, which gives a different absolute level (0.271 vs 0.134 for CD) but the same ranking — treat it as directional.

---

## Repository Structure

```
UCAirfoils/
├── README.md
└── pipeline/
    ├── pipeline_config.yaml
    ├── UCAirfoils.pipeline              ← Elyra visual pipeline graph
    ├── SF_1 … SF_10 notebooks
    ├── metadata/                        ← SF_1…SF_10 YAML configuration
    └── python_nodes_library/
        ├── data_acquisition/
        │   ├── doe.py                   ← define_doe (Sobol) + launch_sim (NeuralFoil)
        │   └── outputs_parser.py
        ├── data_cleansing/
        │   ├── confidence.py            ← analysis_confidence filter
        │   └── manage_missing.py
        ├── feature_selection/
        ├── model_training/
        ├── model_deployment/
        ├── model_validation/
        ├── reporting/
        ├── storage/
        └── tracking/
```
