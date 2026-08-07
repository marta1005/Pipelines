import numpy as np
import pandas as pd
import surrogate_factory as sf

MAX_SPLIT_VAL_ROWS = 3000  # pairwise distance matrix: n² × 8 bytes → 3000² = ~72 MB


@sf.node
def split_validation(workflow, Train_set, Test_set):
    """Validate train/test split quality using voxel tesselation proximity method."""
    from validationlib.misc.split_validation import voxel_tesselation_proximity_method

    inputs = workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'inputs'])

    train_vals = Train_set[inputs].values
    test_vals  = Test_set[inputs].values

    # The method builds an O(n²) pairwise distance matrix. Subsample large datasets
    # to avoid OOM — the split quality check is statistical, so a representative
    # sample is sufficient.
    rng = np.random.default_rng(42)
    if len(train_vals) > MAX_SPLIT_VAL_ROWS:
        idx = rng.choice(len(train_vals), MAX_SPLIT_VAL_ROWS, replace=False)
        train_vals = train_vals[idx]
        print(f"  [split_val] Subsampled train: {len(idx)}/{Train_set.shape[0]} rows")
    if len(test_vals) > MAX_SPLIT_VAL_ROWS:
        idx = rng.choice(len(test_vals), MAX_SPLIT_VAL_ROWS, replace=False)
        test_vals = test_vals[idx]
        print(f"  [split_val] Subsampled test : {len(idx)}/{Test_set.shape[0]} rows")

    result = voxel_tesselation_proximity_method(
        train_vals,
        test_vals,
        categorical_variables=[],
        verbose=False,
    )

    def flag(val, threshold, mode='lt'):
        ok = val <= threshold if mode == 'lt' else val >= threshold
        return '✓' if ok else '✗'

    print("\n  Metric                       Value    Status  (threshold)")
    print("  " + "-" * 58)
    print(f"  Residual voxel proportion  : {result.residual_voxel_proportion:6.3f}   "
          f"{flag(result.residual_voxel_proportion, 0.05)}   (≤ 0.05)")
    print(f"  Valid test proportion      : {result.valid_test_proportion:6.3f}")
    print(f"  Phacking test proportion   : {result.phacking_test_proportion:6.3f}")
    print(f"  Isolated test proportion   : {result.isolated_test_proportion:6.3f}")
    print(f"  Isolated train proportion  : {result.isolated_train_proportion:6.3f}")
    if result.chi_squared_pvalue is not None:
        print(f"  Chi² p-value               : {result.chi_squared_pvalue:6.4f}   "
              f"{flag(result.chi_squared_pvalue, 0.05, 'gt')}   (≥ 0.05)")

    summary = {
        'residual_voxel_proportion': float(result.residual_voxel_proportion),
        'valid_test_proportion':     float(result.valid_test_proportion),
        'phacking_test_proportion':  float(result.phacking_test_proportion),
        'isolated_test_proportion':  float(result.isolated_test_proportion),
        'isolated_train_proportion': float(result.isolated_train_proportion),
        'chi_squared_pvalue': (float(result.chi_squared_pvalue)
                               if result.chi_squared_pvalue is not None else None),
    }
    workflow.metadata.update_step_data({'split_validation': summary})
    return result
