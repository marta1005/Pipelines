import numpy as np
import pandas as pd
import surrogate_factory as sf
from surrogate_factory.catalog.model_validation.metrics import compute_metrics


@sf.node
def calculate_metrics(workflow, Test_set, model_output):
    """Compute R², MAE, quantile90 per output for each trained model."""
    outputs = workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'outputs'])
    models_info = workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models'])
    labels = [m['label'] for m in models_info]

    all_metrics = {label: {} for label in labels}

    col_w = 14
    header = f"\n{'Output':<28}" + "".join(f"{'R² ' + l:>{col_w}}" for l in labels)
    print(header)
    print("-" * (28 + col_w * len(labels)))

    for col in outputs:
        y_test = Test_set[[col]]
        row = f"{col:<28}"
        for label in labels:
            y_pred = model_output[[f"{label}__{col}"]]
            metrics = compute_metrics(y_pred, y_test)
            metrics_clean = {k: float(v) if v is not None else None for k, v in metrics.items()}
            all_metrics[label][col] = metrics_clean
            r2 = metrics_clean.get('R2')
            row += f"{r2:{col_w}.4f}" if r2 is not None else f"{'?':>{col_w}}"
        print(row)

    print("\nBest model per output (by R²):")
    for col in outputs:
        r2s = {label: (all_metrics[label][col].get('R2') or float('-inf')) for label in labels}
        winner = max(r2s, key=r2s.get)
        print(f"  {col:<28} → {winner}  (R²={r2s[winner]:.4f})")

    workflow.metadata.update_step_data({'scores': all_metrics})
    return all_metrics


@sf.node
def distribution_tests(workflow, Train_set, Test_set, train_output, model_output):
    """KS test comparing train vs test residual distributions per model and output."""
    from validationlib.tests.dist import dist_similarity

    outputs = workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'outputs'])
    models_info = workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models'])
    labels = [m['label'] for m in models_info]

    col_w = 12
    all_ks = {}

    for label in labels:
        print(f"\n  KS test p-values — {label}  (H₀: train and test residuals same distribution)")
        print(f"  {'Output':<28}" + f"{'p-value':>{col_w}}" + f"{'status':>{col_w}}")
        print("  " + "-" * (28 + 2 * col_w))

        all_ks[label] = {}
        for col in outputs:
            train_res = (Train_set[col].values
                         - train_output[f"{label}__{col}"].values)
            test_res  = (Test_set[col].values
                         - model_output[f"{label}__{col}"].values)

            _, pval = dist_similarity(train_res, test_res, test='KS')
            status = '✓' if pval >= 0.05 else '✗'
            all_ks[label][col] = float(pval)
            print(f"  {col:<28}{pval:{col_w}.4f}{status:>{col_w}}")

    workflow.metadata.update_step_data({'distribution_tests': all_ks})
    return all_ks
