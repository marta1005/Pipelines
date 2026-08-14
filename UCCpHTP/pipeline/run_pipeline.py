"""
UCCpHTP Surrogate Factory — Standalone Pipeline Runner
=======================================================
Trains an MLP surrogate for the pressure coefficient Cp on a horizontal
tail plane (HTP) from pre-split CFD data.

Inputs  : x, y, z (3D coordinates), alpha (angle of attack), mach
Output  : Cp (pressure coefficient)

IMPORTANT: Provide the pre-split CSV files in UCCpHTP/data/:
    x_train.csv   x_val.csv   x_test.csv    — input columns
    yt_train.csv  yt_val.csv  yt_test.csv   — true output (Cp)

Optional (not used by this pipeline):
    yh_train.csv  yh_val.csv  yh_test.csv   — external model predictions

Usage:
    MLFLOW_ALLOW_FILE_STORE=true python UCCpHTP/pipeline/run_pipeline.py

For the production run (more iterations):
    1. Edit metadata/SF_6_Model_Selection.yaml: max_iter: 500 → max_iter: 2000
    2. Delete the data/artifacts/ folder
    3. Run this script again
"""

import os
import sys
import yaml
import shutil
import pandas as pd
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


def clear_artifacts(artifacts_folder: Path):
    for pattern in ('*.modl', '*.pkl', '*.onnx', '*.json'):
        for f in artifacts_folder.glob(pattern):
            f.unlink()


# ── main ──────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  UCCpHTP — Surrogate Factory Pipeline")
print("=" * 65)

# ── Pre-flight: check required split files ─────────────────────────────────
print("\nInitialising workflow …")
wf = Workflow('pipeline_config.yaml')

split_dir = Path(wf.config['data_split_folder'])
required_files = [
    'x_train.csv', 'x_val.csv', 'x_test.csv',
    'yt_train.csv', 'yt_val.csv', 'yt_test.csv',
]
missing = [f for f in required_files if not (split_dir / f).exists()]
if missing:
    print(f"\nERROR: Missing data files in {split_dir}:")
    for f in missing:
        print(f"  - {f}")
    print("\nExpected files:")
    print("  x_train.csv   x_val.csv   x_test.csv    ← input columns (x, y, z, alpha, mach)")
    print("  yt_train.csv  yt_val.csv  yt_test.csv   ← true output (Cp)")
    sys.exit(1)

load_stage(wf, 'SF_1_Requirements')

data_folder = Path(wf.config['data.folder'])
data_folder.mkdir(parents=True, exist_ok=True)
artifacts_folder = Path(wf.config['artifacts.folder'])
artifacts_folder.mkdir(parents=True, exist_ok=True)

clear_artifacts(artifacts_folder)
job = wf.config['job_name']

# =============================================================================
# Stage 2 — Data Acquisition (load pre-split files)
# =============================================================================
print("\n=== Stage 2: Data Acquisition (pre-split) ===")
load_stage(wf, 'SF_2_Data_Acquisition_Generation')

import re

# Names the rest of the pipeline uses (SF_5 / SF_6 metadata). The pre-split CSVs
# come straight from the CFD export, so neither their headers nor their layout
# are guaranteed: observed in practice are ';' separators and files with no
# header row at all, where the first data line would otherwise be read as the
# column names.
INPUTS  = ['x', 'y', 'z', 'alpha', 'mach']
OUTPUTS = ['Cp']


def _sniff(path):
    """Work out the delimiter, and whether the first line is a header."""
    with open(path) as fh:
        first = fh.readline().rstrip('\r\n')

    counts = {s: first.count(s) for s in (';', ',', '\t', '|')}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        sep = ','                       # single column: any separator will do

    def numeric(tok):
        try:
            float(tok.strip().replace(',', '.') if sep != ',' else tok.strip())
            return True
        except ValueError:
            return False

    fields = [f for f in first.split(sep) if f.strip() != '']
    # All-numeric first line means the export wrote no header.
    header = None if fields and all(numeric(f) for f in fields) else 0
    return sep, header


def load_split(filename, expected, what):
    """Read a split CSV and return it with the pipeline's column names."""
    path = split_dir / filename
    sep, header = _sniff(path)
    df = pd.read_csv(path, sep=sep, header=header)
    if header is None:
        print(f"  {filename}: no header row, sep={sep!r} -> naming columns {expected}")
    elif sep != ',':
        print(f"  {filename}: sep={sep!r}")

    junk = [c for c in df.columns if re.fullmatch(r'Unnamed: \d+', str(c))]
    if junk:
        df = df.drop(columns=junk)
        print(f"  {filename}: dropped index column(s) {junk}")

    # With no header an index column has no name to recognise it by, so spot it
    # by shape: one column too many, the first being a consecutive integer run.
    if header is None and len(df.columns) == len(expected) + 1:
        first = df.iloc[:, 0]
        step = first.diff().dropna()
        if (first % 1 == 0).all() and (step == 1).all():
            df = df.iloc[:, 1:]
            print(f"  {filename}: dropped unnamed positional index column")

    if len(df.columns) != len(expected):
        raise ValueError(
            f"{filename}: expected {len(expected)} {what} column(s) {expected}, "
            f"but the file has {len(df.columns)}: {list(df.columns)[:8]}.\n"
            f"Detected separator {sep!r}, header={'yes' if header == 0 else 'no'}. "
            f"Check the file, or edit INPUTS/OUTPUTS to match your export."
        )

    if list(df.columns) != expected:
        if header is None:
            # No names in the file: position is the only information there is.
            df = df.set_axis(expected, axis=1)
        else:
            # The file does name its columns, so match on the name rather than
            # the position — renaming positionally would silently mislabel a
            # file whose columns are in a different order.
            norm = {str(c).strip().lower().lstrip('﻿'): c for c in df.columns}
            matched = {e: norm.get(e.lower()) for e in expected}
            if all(v is not None for v in matched.values()):
                df = df[[matched[e] for e in expected]].set_axis(expected, axis=1)
                print(f"  {filename}: matched columns by name -> {expected}")
            else:
                unmatched = [e for e, v in matched.items() if v is None]
                print(f"  {filename}: WARNING no column named {unmatched} in "
                      f"{list(df.columns)}; falling back to position "
                      f"{list(df.columns)} -> {expected}. Check the order is right.")
                df = df.set_axis(expected, axis=1)

    return df.apply(pd.to_numeric, errors='coerce')


x_train = load_split('x_train.csv', INPUTS, 'input')
x_val   = load_split('x_val.csv',   INPUTS, 'input')
x_test  = load_split('x_test.csv',  INPUTS, 'input')

yt_train = load_split('yt_train.csv', OUTPUTS, 'output')
yt_val   = load_split('yt_val.csv',   OUTPUTS, 'output')
yt_test  = load_split('yt_test.csv',  OUTPUTS, 'output')

Train_set = pd.concat([x_train.reset_index(drop=True),
                        yt_train.reset_index(drop=True)], axis=1)
Val_set   = pd.concat([x_val.reset_index(drop=True),
                        yt_val.reset_index(drop=True)], axis=1)
Test_set  = pd.concat([x_test.reset_index(drop=True),
                        yt_test.reset_index(drop=True)], axis=1)

print(f"  Train : {Train_set.shape[0]:>8,} rows   columns: {list(Train_set.columns)}")
print(f"  Val   : {Val_set.shape[0]:>8,} rows")
print(f"  Test  : {Test_set.shape[0]:>8,} rows")

wf.save_data(Train_set, f"{job}_Train_set.csv")
wf.save_data(Val_set,   f"{job}_Val_set.csv")
wf.save_data(Test_set,  f"{job}_Test_set.csv")

# =============================================================================
# Stage 3 — Data Cleansing (skip — data is already clean)
# =============================================================================
print("\n=== Stage 3: Data Cleansing (skipped — pre-split data assumed clean) ===")
load_stage(wf, 'SF_3_Data_Cleansing')
nulls = Train_set.isnull().sum()
remaining = nulls[nulls > 0]
if not remaining.empty:
    print(f"WARNING: Nulls found in Train_set:\n{remaining}")
else:
    print("  No missing values.")

# =============================================================================
# Stage 4 — Data Partitioning (skip — already split)
# =============================================================================
print("\n=== Stage 4: Data Partitioning (skipped — files are already split) ===")
load_stage(wf, 'SF_4_Data_Partitioning')

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
print(f"  Metadata  : {artifacts_folder}/metadata_{job}.json")

# =============================================================================
# Reporting
# =============================================================================
print("\n=== Reporting: Generate Executive Summary ===")
import subprocess

meta_json = artifacts_folder / f"metadata_{job}.json"
report_dir = artifacts_folder / 'validation_reports'
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
    2. Delete the data/artifacts/ folder
    3. MLFLOW_ALLOW_FILE_STORE=true python run_pipeline.py
""")
