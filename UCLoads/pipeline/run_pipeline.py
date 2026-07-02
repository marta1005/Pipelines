"""
UCLoads Surrogate Factory — Standalone Pipeline Runner
=======================================================
Runs all pipeline stages without Jupyter.

Usage (from Pipelines root or pipeline directory):
    python UCLoads/pipeline/run_pipeline.py

For a production run with real data:
    1. Replace datasets/TrainData/ with real .mon files
    2. Delete the data/ folder (or its .csv and .json files)
    3. Run this script again
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

# Disable OpenMP multi-threading to prevent XGBoost segfault on macOS
os.environ.setdefault('OMP_NUM_THREADS', '1')

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT    = PIPELINE_DIR.parent.parent  # Desktop/Pipelines

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(PIPELINE_DIR / "python_nodes_library"))
sys.path.insert(0, str(REPO_ROOT))

os.chdir(PIPELINE_DIR)

from surrogate_factory.workflow import Workflow


def load_stage(workflow, stage_name):
    """Load a stage YAML directly (no Jupyter widgets required)."""
    yaml_path = Path(workflow.config['metadata.folder']) / f"{stage_name}.yaml"
    with open(yaml_path, 'r') as f:
        new_data = yaml.safe_load(f)
    workflow.metadata.update_step_data(new_data, ["metadata"])


def clear_data_folder(data_folder: Path):
    """Remove artefacts from a previous run so save_data() does not skip."""
    for pattern in ('*.csv', '*.json', '*.modl', '*.pkl'):
        for f in data_folder.glob(pattern):
            f.unlink()
    artifacts = data_folder / 'artifacts'
    if artifacts.exists():
        for pattern in ('*.modl', '*.pkl', '*.json'):
            for f in artifacts.glob(pattern):
                f.unlink()


# ── Initialise ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  UCLoads — Surrogate Factory Pipeline")
print("=" * 65)

wf = Workflow('pipeline_config.yaml')

data_folder = Path(wf.config['data.folder'])
data_folder.mkdir(parents=True, exist_ok=True)
(data_folder / 'artifacts').mkdir(parents=True, exist_ok=True)
clear_data_folder(data_folder)

# Load SF_1 requirements so validate() can find them in Stage 9
load_stage(wf, 'SF_1_Requirements')
job = wf.config['job_name']

# =============================================================================
# Stage 2 — Data Acquisition
# =============================================================================
print("\n=== Stage 2: Data Acquisition ===")
load_stage(wf, 'SF_2_Data_Acquisition_Generation')

from data_acquisition.load_mon import batch_extract, batch_transform, batch_load

raw_data    = batch_extract(wf)
transformed = batch_transform(wf, raw_data)
Dataset     = batch_load(wf, transformed)
wf.save_data(Dataset, f"{job}_raw_dataset.csv")

# =============================================================================
# Stage 3 — Data Cleansing
# =============================================================================
print("\n=== Stage 3: Data Cleansing ===")
load_stage(wf, 'SF_3_Data_Cleansing')

from data_cleansing.clean import drop_missing, report_statistics

Dataset = drop_missing(wf, Dataset)
Dataset = report_statistics(wf, Dataset)
wf.save_data(Dataset, f"{job}_clean_dataset.csv")

# =============================================================================
# Stage 4 — Data Partitioning
# =============================================================================
print("\n=== Stage 4: Data Partitioning ===")
load_stage(wf, 'SF_4_Data_Partitioning')

from data_partition.partition import data_split

Train_set, Test_set, Val_set = data_split(wf, Dataset)
wf.save_data(Train_set, f"{job}_Train_set.csv")
wf.save_data(Val_set,   f"{job}_Val_set.csv")
wf.save_data(Test_set,  f"{job}_Test_set.csv")

# =============================================================================
# Stage 5 — Feature Selection
# =============================================================================
print("\n=== Stage 5: Feature Selection ===")
load_stage(wf, 'SF_5_Feature_Selection')

from feature_selection.preprocess import preprocessor

Train_set = wf.load_data(f"{job}_Train_set.csv")
preprocessor(wf, Train_set)

# =============================================================================
# Stage 6 — Model Selection  (config only)
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
models_info = train(wf, Train_set, Val_set)

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
split_result = split_validation(wf, Train_set, Test_set)

print("\n--- 9.1 Predictions ---")
model_output = predict(wf, Test_set)
train_output = predict(wf, Train_set)
val_output   = predict(wf, Val_set)

print("\n--- 9.2 Metrics ---")
metrics = calculate_metrics(wf, Test_set, model_output)

print("\n--- 9.2b Distribution Tests ---")
ks_results = distribution_tests(wf, Train_set, Test_set, train_output, model_output)

print("\n--- 9.3 Validation against requirements ---")
validate(wf, metrics)

print("\n--- 9.4 Plots ---")
import matplotlib
matplotlib.use('Agg')  # headless — no display needed
plot(wf, Test_set, model_output)

print("\n--- 9.5 Validation Reports ---")
csv_dirs = export_validation_csvs(
    wf, Train_set, Val_set, Test_set,
    train_output, val_output, model_output,
)

LIB_DIR    = PIPELINE_DIR / "python_nodes_library"
script_path = LIB_DIR / "model_validation" / "validation_script.py"
output_dir  = Path(wf.config["artifacts.folder"]) / "validation_reports"
output_dir.mkdir(exist_ok=True)

ms = wf.metadata.get_step_data(['metadata', 'Model_Selection'])
num_inputs = ms['inputs']

for label, csv_dir in csv_dirs.items():
    print(f"\n  Running report for {label}...")
    cmd = [
        sys.executable, str(script_path),
        "-d", str(csv_dir),
        "-n", label,
        "-o", str(output_dir),
        "--exclude_warnings",
        "--splitting_variables", *num_inputs,
    ]
    env = {**os.environ, "MLFLOW_ALLOW_FILE_STORE": "true"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        report = output_dir / f"{label}_validation_output.html"
        print(f"  ✓ {report}")
        if result.stdout:
            print(result.stdout[-400:])
    else:
        debug_file = output_dir / f"{label}_error.txt"
        debug_file.write_text(result.stdout + "\n" + result.stderr)
        print(f"  ✗ FAILED — see {debug_file}")
        print("STDERR:", result.stderr[-800:])

print("\n--- 9.6 MLflow EDA Report ---")
from tracking.mlflow_eda import log_eda

log_eda(
    workflow=wf,
    Train_set=Train_set, Val_set=Val_set, Test_set=Test_set,
    model_output=model_output, train_output=train_output,
    metrics=metrics, split_result=split_result, ks_results=ks_results,
)

# =============================================================================
# Save metadata
# =============================================================================
wf.save_metadata()

print("\n" + "=" * 65)
print("  Pipeline complete!")
print("=" * 65)
print(f"\n  Artifacts : {wf.config['artifacts.folder']}")
print(f"  Metadata  : {wf.config['artifacts.folder']}/metadata_{job}.json")
print(f"  Reports   : {output_dir}")
