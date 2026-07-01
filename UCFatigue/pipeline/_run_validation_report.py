"""Standalone script to generate validation HTML reports for all UCFatigue models."""
import os
import sys
import subprocess
from pathlib import Path

# ── Setup paths ──────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
LIB_DIR = PIPELINE_DIR / "python_nodes_library"
REPO_ROOT = PIPELINE_DIR.parent.parent  # Desktop/Pipelines

# Add library dirs to path so sf + model_validation imports work
sys.path.insert(0, str(REPO_ROOT / "src"))       # surrogate_factory
sys.path.insert(0, str(LIB_DIR))                  # model_validation, tracking, etc.

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── Import after path setup ───────────────────────────────────────────────────
os.chdir(DATA_DIR)  # workflow.resume() expects this
from surrogate_factory.workflow import Workflow
from model_validation.prediction import predict
from model_validation.export_validation_csvs import export_validation_csvs

# ── Load workflow ─────────────────────────────────────────────────────────────
os.chdir(PIPELINE_DIR)  # config file is here
workflow = Workflow("pipeline_config.yaml")
workflow.resume()

# ── Load datasets ─────────────────────────────────────────────────────────────
job = workflow.config["job_name"]
Train_set = workflow.load_data(f"{job}_Train_set.csv")
Val_set   = workflow.load_data(f"{job}_Val_set.csv")
Test_set  = workflow.load_data(f"{job}_Test_set.csv")
print(f"Train: {Train_set.shape}  Val: {Val_set.shape}  Test: {Test_set.shape}")

# ── Predictions ───────────────────────────────────────────────────────────────
print("Running predictions...")
model_output = predict(workflow, Test_set)
train_output = predict(workflow, Train_set)
val_output   = predict(workflow, Val_set)

# ── Export CSVs ───────────────────────────────────────────────────────────────
print("Exporting validation CSVs...")
csv_dirs = export_validation_csvs(
    workflow,
    Train_set, Val_set, Test_set,
    train_output, val_output, model_output,
)

# ── Run validation_script.py for each model ───────────────────────────────────
script_path = LIB_DIR / "model_validation" / "validation_script.py"
output_dir  = Path(workflow.config["artifacts.folder"]) / "validation_reports"
output_dir.mkdir(exist_ok=True)

ms = workflow.metadata.get_step_data(["metadata", "Model_Selection"])
num_inputs = Train_set[ms["inputs"]].select_dtypes(include="number").columns.tolist()

for label, csv_dir in csv_dirs.items():
    print(f"\n{'='*60}")
    print(f"Running validation report for: {label}")
    cmd = [
        sys.executable, str(script_path),
        "-d", str(csv_dir),
        "-n", label,
        "-o", str(output_dir),
        "--exclude_warnings",
        "--splitting_variables", *num_inputs,
    ]
    env = {
        **os.environ,
        "PYTHONPATH": ":".join([str(REPO_ROOT / "src"), str(LIB_DIR)]),
        "MLFLOW_ALLOW_FILE_STORE": "true",
    }
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        report = output_dir / f"{label}_validation_output.html"
        print(f"  ✓ Report saved: {report}")
        if result.stdout:
            print(result.stdout[-400:])
    else:
        print(f"  ✗ FAILED (return code {result.returncode})")
        if result.stdout:
            # Write full stdout to a debug file
            debug_file = output_dir / f"{label}_error.txt"
            debug_file.write_text(result.stdout)
            print(f"  Full error written to: {debug_file}")
            # Show last 1200 chars
            print("STDOUT (last 1200):", result.stdout[-1200:])
        if result.stderr:
            print("STDERR:", result.stderr[-300:])

print("\nDone.")
