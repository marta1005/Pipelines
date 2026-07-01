"""
Export CSVs for the validation_template.ipynb report.

Creates one folder per model under artifacts/validation_{label}/ with:
  x_train.csv  x_val.csv  x_test.csv   — numeric inputs (no categorical)
  yt_train.csv yt_val.csv yt_test.csv  — true outputs
  yh_val.csv   yh_test.csv             — predicted outputs for that model
"""

import pandas as pd
from pathlib import Path


def export_validation_csvs(workflow,
                            Train_set, Val_set, Test_set,
                            train_output, val_output, model_output):
    """
    Export all CSVs required by validation_template.ipynb, one folder per model.

    Parameters
    ----------
    train_output : DataFrame
        Predictions on Train_set  (columns '{label}__{col}').
    val_output   : DataFrame
        Predictions on Val_set    (columns '{label}__{col}').
    model_output : DataFrame
        Predictions on Test_set   (columns '{label}__{col}').

    Returns
    -------
    dict  {label: Path to csv_dir}
    """
    ms = workflow.metadata.get_step_data(['metadata', 'Model_Selection'])
    inputs  = ms['inputs']
    outputs = list(ms['outputs'])
    models_info = workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models'])

    artifacts = Path(workflow.config['artifacts.folder'])
    num_inputs = Train_set[inputs].select_dtypes(include='number').columns.tolist()

    dirs = {}
    for info in models_info:
        label = info['label']
        csv_dir = artifacts / f'validation_{label}'
        csv_dir.mkdir(parents=True, exist_ok=True)

        # ── Inputs (numeric only — validationlib plots require numeric) ──────
        Train_set[num_inputs].reset_index(drop=True).to_csv(
            csv_dir / 'x_train.csv', index=False)
        Val_set[num_inputs].reset_index(drop=True).to_csv(
            csv_dir / 'x_val.csv',   index=False)
        Test_set[num_inputs].reset_index(drop=True).to_csv(
            csv_dir / 'x_test.csv',  index=False)

        # ── True outputs ──────────────────────────────────────────────────────
        Train_set[outputs].reset_index(drop=True).to_csv(
            csv_dir / 'yt_train.csv', index=False)
        Val_set[outputs].reset_index(drop=True).to_csv(
            csv_dir / 'yt_val.csv',   index=False)
        Test_set[outputs].reset_index(drop=True).to_csv(
            csv_dir / 'yt_test.csv',  index=False)

        # ── Predictions (rename '{label}__{col}' → '{col}') ──────────────────
        pd.DataFrame(
            {col: val_output[f'{label}__{col}'].values   for col in outputs}
        ).to_csv(csv_dir / 'yh_val.csv',  index=False)

        pd.DataFrame(
            {col: model_output[f'{label}__{col}'].values for col in outputs}
        ).to_csv(csv_dir / 'yh_test.csv', index=False)

        dirs[label] = csv_dir
        print(f'  [{label}] CSVs → {csv_dir}')

    return dirs
