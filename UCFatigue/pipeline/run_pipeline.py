"""
UCFatigue Surrogate Factory — Standalone Pipeline Runner
=========================================================
Runs all pipeline stages without Jupyter.

Usage (from project root or pipeline directory):
    python UCFatigue/pipeline/run_pipeline.py

For the production run (more epochs):
    1. Edit metadata/SF_6_Model_Selection.yaml: max_iter: 5 → max_iter: 500
    2. Delete the data/ folder (or its .csv and .json files)
    3. Run this script again
"""

import os
import sys
import yaml
import shutil
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
SRC_DIR = PIPELINE_DIR.parent.parent / 'src'
REPO_ROOT = PIPELINE_DIR.parent.parent  # for validationlib

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PIPELINE_DIR / 'python_nodes_library'))
sys.path.insert(0, str(REPO_ROOT))

# Workflow expects to be initialised from a directory that contains
# pipeline_config.yaml, but set_paths() will cd to data.folder.
os.chdir(PIPELINE_DIR)

from surrogate_factory.workflow import Workflow


# ── helpers ───────────────────────────────────────────────────────────────────

def load_stage(workflow, stage_name):
    """Load a stage YAML into workflow metadata without triggering Jupyter widgets."""
    meta_folder = Path(workflow.config['metadata.folder'])
    yaml_path = meta_folder / f"{stage_name}.yaml"
    with open(yaml_path, 'r') as f:
        new_data = yaml.safe_load(f)
    workflow.metadata.update_step_data(new_data, ["metadata"])


def clear_data_folder(data_folder: Path):
    """Remove all CSVs and JSONs from a previous run so save_data() does not skip them."""
    for pattern in ('*.csv', '*.json', '*.modl', '*.pkl', '*.onnx'):
        for f in data_folder.glob(pattern):
            f.unlink()
    artifacts = data_folder / 'artifacts'
    if artifacts.exists():
        for pattern in ('*.modl', '*.pkl', '*.onnx', '*.json'):
            for f in artifacts.glob(pattern):
                f.unlink()


# ── main ──────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  UCFatigue — Surrogate Factory Pipeline")
print("=" * 65)

# --- init workflow -----------------------------------------------------------
print("\nInitialising workflow …")
wf = Workflow('pipeline_config.yaml')

data_folder = Path(wf.config['data.folder'])
data_folder.mkdir(parents=True, exist_ok=True)
(data_folder / 'artifacts').mkdir(parents=True, exist_ok=True)

clear_data_folder(data_folder)
job = wf.config['job_name']

# =============================================================================
# Stage 2 — Data Acquisition
# =============================================================================
print("\n=== Stage 2: Data Acquisition ===")
load_stage(wf, 'SF_2_Data_Acquisition_Generation')

from data_acquisition.outputs_parser import batch_extract, batch_transform

dataset = batch_extract(wf)
dataset = batch_transform(wf, dataset)
print(f"Shape after extract+transform: {dataset.shape}")
wf.save_data(dataset, f"{job}_Raw.csv")

# =============================================================================
# Stage 3 — Data Cleansing
# =============================================================================
print("\n=== Stage 3: Data Cleansing ===")
load_stage(wf, 'SF_3_Data_Cleansing')

from data_cleansing.manage_missing import replace_missing_values

dataset = wf.load_data(f"{job}_Raw.csv")
dataset_clean = replace_missing_values(wf, dataset)
nulls = dataset_clean.isnull().sum()
remaining = nulls[nulls > 0]
if remaining.empty:
    print("No missing values remaining.")
else:
    print(f"Remaining nulls:\n{remaining}")
wf.save_data(dataset_clean, f"{job}_Cleaned.csv")

# =============================================================================
# Stage 4 — Data Partitioning
# =============================================================================
print("\n=== Stage 4: Data Partitioning ===")
load_stage(wf, 'SF_4_Data_Partitioning')

from sklearn.model_selection import train_test_split

dataset = wf.load_data(f"{job}_Cleaned.csv")
Train_set, Test_set = train_test_split(dataset, test_size=0.2, random_state=42)
Train_set, Val_set  = train_test_split(Train_set, test_size=0.125, random_state=42)
# Resulting splits: ~70 % train / ~10 % val / ~20 % test

print(f"Train : {Train_set.shape[0]} rows")
print(f"Val   : {Val_set.shape[0]} rows")
print(f"Test  : {Test_set.shape[0]} rows")

wf.save_data(Train_set, f"{job}_Train_set.csv")
wf.save_data(Val_set,   f"{job}_Val_set.csv")
wf.save_data(Test_set,  f"{job}_Test_set.csv")

# =============================================================================
# Stage 5 — Feature Selection & Preprocessing
# =============================================================================
print("\n=== Stage 5: Feature Selection & Preprocessing ===")
load_stage(wf, 'SF_5_Feature_Selection')

from feature_selection.preprocess import preprocessor

Train_set = wf.load_data(f"{job}_Train_set.csv")
preprocessor(wf, Train_set)

# =============================================================================
# Stage 6 — Model Selection  (config only, no computation)
# =============================================================================
print("\n=== Stage 6: Model Selection ===")
load_stage(wf, 'SF_6_Model_Selection')
ms = wf.metadata.get_step_data(['metadata', 'Model_Selection'])
for alg in ms['algorithms']:
    print(f"  [{alg['label']}] {alg['name']}  settings: {alg['settings']}")

# =============================================================================
# Stage 7 — Model Training
# =============================================================================
print("\n=== Stage 7: Model Training ===")
load_stage(wf, 'SF_7_Model_Training')

from model_training.learn import train

Train_set = wf.load_data(f"{job}_Train_set.csv")
Val_set   = wf.load_data(f"{job}_Val_set.csv")
model = train(wf, Train_set, Val_set)

# =============================================================================
# Stage 8 — Model Deployment
# =============================================================================
print("\n=== Stage 8: Model Deployment ===")
load_stage(wf, 'SF_8_Model_Deployment')

from model_deployment.model import model_deployment

model_deployment(wf)

# =============================================================================
# Stage 9 — Model Validation
# =============================================================================
print("\n=== Stage 9: Model Validation ===")
load_stage(wf, 'SF_9_Model_Validation')

from model_validation.split_val import split_validation
from model_validation.prediction import predict
from model_validation.score import calculate_metrics, distribution_tests
from model_validation.validation import validate
from model_validation.visualize import plot

Train_set = wf.load_data(f"{job}_Train_set.csv")
Test_set  = wf.load_data(f"{job}_Test_set.csv")

# 9.0 — split quality
print("\n--- 9.0 Split Validation ---")
split_validation(wf, Train_set, Test_set)

# 9.1 — predictions on test set (and train set for distribution tests)
print("\n--- 9.1 Predictions ---")
model_output = predict(wf, Test_set)
train_output = predict(wf, Train_set)

# 9.2 — R² / MAE metrics
print("\n--- 9.2 Metrics ---")
metrics = calculate_metrics(wf, Test_set, model_output)

# 9.2b — KS distribution tests on residuals
print("\n--- 9.2b Distribution Tests (KS: train vs test residuals) ---")
distribution_tests(wf, Train_set, Test_set, train_output, model_output)

# 9.3 — validation against requirements
print("\n--- 9.3 Validation against requirements ---")
validate(wf, metrics)

# 9.4 — plots
print("\n--- 9.4 Plots ---")
plot(wf, Test_set, model_output)

# =============================================================================
# Save metadata and print summary
# =============================================================================
wf.save_metadata()

print("\n" + "=" * 65)
print("  Pipeline complete!")
print("=" * 65)
print(f"\n  Artifacts : {wf.config['artifacts.folder']}")
print(f"  Metadata  : {wf.config['artifacts.folder']}/metadata_{job}.json")
print("""
  For the production run (500 iterations):
    1. Edit metadata/SF_6_Model_Selection.yaml
         max_iter: 5  →  max_iter: 500
    2. Delete the data/ folder
    3. python run_pipeline.py
""")
