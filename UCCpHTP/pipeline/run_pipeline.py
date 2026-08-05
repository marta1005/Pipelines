"""
UCCpHTP Surrogate Factory — Standalone Pipeline Runner
=======================================================
Trains an MLP surrogate for the pressure coefficient Cp on a horizontal
tail plane (HTP) from CFD data.

Inputs  : x, y, z (3D coordinates), alpha (angle of attack), mach
Output  : Cp (pressure coefficient)

IMPORTANT: Provide the CFD dataset at UCCpHTP/data/cphtp_data.csv
           Expected columns (comma-separated): x, y, z, alpha, mach, Cp

Usage:
    MLFLOW_ALLOW_FILE_STORE=true python UCCpHTP/pipeline/run_pipeline.py

For the production run (more iterations):
    1. Edit metadata/SF_6_Model_Selection.yaml: max_iter: 500 → max_iter: 2000
    2. Delete the data/ folder
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
REPO_ROOT = PIPELINE_DIR.parent.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PIPELINE_DIR / 'python_nodes_library'))
sys.path.insert(0, str(REPO_ROOT))

os.chdir(PIPELINE_DIR)

from surrogate_factory.workflow import Workflow


# ── helpers ───────────────────────────────────────────────────────────────────

def load_stage(workflow, stage_name):
    meta_folder = Path(workflow.config['metadata.folder'])
    yaml_path = meta_folder / f"{stage_name}.yaml"
    with open(yaml_path, 'r') as f:
        new_data = yaml.safe_load(f)
    workflow.metadata.update_step_data(new_data, ["metadata"])


def clear_data_folder(data_folder: Path):
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
print("  UCCpHTP — Surrogate Factory Pipeline")
print("=" * 65)

# Check data file exists
input_csv = Path('/Users/martaarnabatmartin/Desktop/Pipelines/UCCpHTP/data/cphtp_data.csv')
if not input_csv.exists():
    print(f"\nERROR: Data file not found: {input_csv}")
    print("Please provide the CFD dataset (columns: x, y, z, alpha, mach, Cp)")
    sys.exit(1)

print("\nInitialising workflow …")
wf = Workflow('pipeline_config.yaml')

load_stage(wf, 'SF_1_Requirements')

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
# Stage 6 — Model Selection
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
from model_validation.export_validation_csvs import export_validation_csvs

Train_set = wf.load_data(f"{job}_Train_set.csv")
Val_set   = wf.load_data(f"{job}_Val_set.csv")
Test_set  = wf.load_data(f"{job}_Test_set.csv")

print("\n--- 9.0 Split Validation ---")
split_validation(wf, Train_set, Test_set)

print("\n--- 9.1 Predictions ---")
model_output = predict(wf, Test_set)
train_output = predict(wf, Train_set)
val_output   = predict(wf, Val_set)

print("\n--- 9.2 Metrics ---")
metrics = calculate_metrics(wf, Test_set, model_output)

print("\n--- 9.2b Distribution Tests ---")
distribution_tests(wf, Train_set, Test_set, train_output, model_output)

print("\n--- 9.3 Validation against requirements ---")
validate(wf, metrics)

print("\n--- 9.4 Plots ---")
plot(wf, Test_set, model_output)

print("\n--- 9.5 Export Validation CSVs ---")
export_validation_csvs(wf, Train_set, Val_set, Test_set,
                        train_output, val_output, model_output)

# =============================================================================
wf.save_metadata()

print("\n" + "=" * 65)
print("  Pipeline complete!")
print("=" * 65)
print(f"\n  Artifacts : {wf.config['artifacts.folder']}")
print(f"  Metadata  : {wf.config['artifacts.folder']}/metadata_{job}.json")

# =============================================================================
# Reporting
# =============================================================================
print("\n=== Reporting: Generate Executive Summary ===")
import subprocess

meta_json = Path(wf.config['artifacts.folder']) / f"metadata_{job}.json"
report_dir = Path(wf.config['artifacts.folder']) / 'validation_reports'
report_dir.mkdir(parents=True, exist_ok=True)

reporting_script = PIPELINE_DIR / 'python_nodes_library' / 'reporting' / 'generate_executive_summary.py'
result = subprocess.run(
    [sys.executable, str(reporting_script), str(meta_json), '--output', str(report_dir)],
    capture_output=False,
)
if result.returncode != 0:
    print("WARNING: Report generation failed. Run manually:")
    print(f"  python {reporting_script} {meta_json} --output {report_dir}")

print("""
  For the production run (more iterations):
    1. Edit metadata/SF_6_Model_Selection.yaml
         max_iter: 500  →  max_iter: 2000
    2. Delete the data/ folder
    3. MLFLOW_ALLOW_FILE_STORE=true python run_pipeline.py
""")
