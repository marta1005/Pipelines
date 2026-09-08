---
name: sf-pipeline
description: How to scaffold, configure, run and validate a Surrogate Factory v2.2 use case (UCxxx) in this repository. Use whenever the task involves creating a new surrogate pipeline, editing SF_N stage YAMLs, running a pipeline, or interpreting its validation results.
---

# Building a Surrogate Factory pipeline

Surrogate Factory (SF v2.2, `src/surrogate_factory/`) structures surrogate development into 9 declarative stages plus an optional storage stage. Every use case (UCFatigue, UCLoads, UCAirfoils, UCHardLanding, UCCpHTP) follows the exact same skeleton. When asked to build a new pipeline, copy the conventions below; never invent a new layout.

## Use-case layout (mandatory)

```
UC<Name>/
├── README.md
├── data/                          # raw input data (CSV or generated)
└── pipeline/
    ├── pipeline_config.yaml       # paths, tracker, catalog
    ├── UC<Name>.pipeline          # Elyra graph (optional, copy and rename)
    ├── run_pipeline.py            # standalone runner (copy from UCHardLanding)
    ├── SF_1_Requirements.ipynb ... SF_9_Model_Validation.ipynb (SF_10 optional)
    ├── metadata/                  # one YAML per stage: SF_1..SF_10
    └── python_nodes_library/      # @sf.node functions per stage
```

The 9 stages: SF_1 Requirements, SF_2 Data Acquisition, SF_3 Data Cleansing, SF_4 Data Partitioning (70/10/20, seed 42), SF_5 Feature Selection (normalizer_transformer), SF_6 Model Selection, SF_7 Model Training (logs to MLflow), SF_8 Model Deployment (scaler+model .pkl), SF_9 Model Validation (metrics, reports), SF_10 Model Storage (optional, integration test).

## pipeline_config.yaml (real example, UCHardLanding)

```yaml
Surrogate Factory: '2.2'
job_name: UCHARDLANDING_1          # UPPERCASE UC name + _1
data.folder: <abs>/UC<Name>/pipeline/data
artifacts.folder: <abs>/UC<Name>/pipeline/data/artifacts
metadata.folder: <abs>/UC<Name>/pipeline/metadata
python_libs.folder: <abs>/UC<Name>/pipeline/python_nodes_library
input_data: <abs>/UC<Name>/data/<file>.csv
tracker:
  tool: mlflow
  options:
    tracking_uri: file://<abs>/UC<Name>/pipeline/mlruns
catalog:
  models:
    MLPRegressor: model_training.val_curve.MLPRegressorValCurve
    MultiOutputGradientBoosting: model_training.estimators.MultiOutputGradientBoosting
  normalization:
    normalizer_transformer: surrogate_factory.catalog.feature_selection.normalization.normalizer_transformer
  metrics:
    compute_metrics: surrogate_factory.catalog.model_validation.metrics.compute_metrics
```

Paths are absolute on the user's machine: derive them from the repo root of the open workspace, never copy another machine's home directory.

## Stage YAML conventions (copy these shapes exactly)

SF_1 requirements, per output:

```yaml
Requirements:
  accuracy:
    - output: "O1"
      metric: quantile90
      value: 0.1
```

SF_2 extraction:

```yaml
Data_Acquisition_Generation:
  Data_Acquisition:
    Extract:
      method: pandas.read_csv
      path: <abs path to csv>
      sep: ";"        # match the actual file
```

SF_6 model selection: `inputs:` list, `outputs:` list, `algorithms:` list of `{name, label, settings}` where `name` must exist in the catalog. Two candidates minimum (champion + baseline). Always `random_state: 42`.

## Model choice rules (from the five validated use cases)

- Small or discrete tabular data (≤100 k rows, categorical levels): MultiOutputGradientBoosting champion, MLP baseline (UCFatigue, UCHardLanding).
- Large smooth continuous fields (≥100 k rows): MLP (256-128-64, ReLU, Adam, early stopping patience 20) champion, GB baseline (UCAirfoils, UCCpHTP).
- Standardise multi-output targets for MLPs, otherwise the widest output captures the shared loss (UCAirfoils lesson).
- Requirements are defined per output BEFORE training; the test set is never used for any training decision.

## Run and verify

1. Smoke run first: set `max_iter: 5` (MLP) or `n_estimators: 20` (GB) in SF_6, run `python UC<Name>/pipeline/run_pipeline.py` (copy the UCHardLanding runner: it inserts `src/`, `python_nodes_library/` and the repo root into `sys.path`, loads each stage YAML, and executes SF_1..SF_9 in order).
2. If the smoke run passes, restore production settings, delete `pipeline/data/*.csv|json` and rerun.
3. Read the outcome, in this order:
   - `pipeline/data/artifacts/validation_reports/executive_summary_UC..._1.tex|pdf`: verdict banner, Q90/R² per output, split quality (KS/AD, VTP), bias matrix, interval coverage.
   - `analysis_plots/`: winner_pred_true.png, err_relerr_cdf.png, split and uncertainty plots.
   - MLflow under `pipeline/mlruns/`.
4. Interpret with the validation playbook: bad split → re-partition, never re-tune; Q90 fails in every model → data coverage; targets near zero → fix the metric in SF_1 (range-normalised), not the model; under-coverage → recalibrate intervals.

## Rules

- Reuse `python_nodes_library` modules from UCHardLanding as the template; only `data_acquisition` parsing usually needs edits.
- Never modify `src/surrogate_factory/` or `validationlib/` when building a use case.
- Never hand-edit files in `pipeline/data/`; they are pipeline outputs.
- Report at the end: requirement verdict per output, champion model, and the next corrective action if anything fails.
