import pandas as pd
import surrogate_factory as sf


@sf.node
def split_validation(workflow, Train_set, Test_set):
    """Validate train/test split quality using voxel tesselation proximity method."""
    from validationlib.misc.split_validation import voxel_tesselation_proximity_method

    inputs = workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'inputs'])

    # FLAP and Type_segment are the categorical features in our input space
    categorical_cols = [inputs.index('FLAP'), inputs.index('Type_segment')]

    # VTP builds an O(n^2) pairwise distance matrix per test point, so it blows up
    # on large sets. Cap the sample; the split check is statistical, so a
    # representative subset is enough. Column indices above are unaffected.
    MAX_VTP_ROWS = 2000
    train_in = Train_set[inputs]
    test_in  = Test_set[inputs]
    if len(train_in) > MAX_VTP_ROWS:
        train_in = train_in.sample(MAX_VTP_ROWS, random_state=42)
        print(f"  [split_val] Subsampled train: {MAX_VTP_ROWS}/{len(Train_set)} rows")
    if len(test_in) > MAX_VTP_ROWS:
        test_in = test_in.sample(MAX_VTP_ROWS, random_state=42)
        print(f"  [split_val] Subsampled test : {MAX_VTP_ROWS}/{len(Test_set)} rows")

    result = voxel_tesselation_proximity_method(
        train_in.values,
        test_in.values,
        categorical_variables=categorical_cols,
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
