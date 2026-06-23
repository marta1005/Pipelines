import pandas as pd
import surrogate_factory as sf


@sf.node
def split_validation(workflow, Train_set, Test_set):
    """Validate train/test split quality using voxel tesselation proximity method."""
    from validationlib.misc.split_validation import voxel_tesselation_proximity_method

    inputs = workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'inputs'])

    # FLAP and Type_segment are the categorical features in our input space
    categorical_cols = [inputs.index('FLAP'), inputs.index('Type_segment')]

    result = voxel_tesselation_proximity_method(
        Train_set[inputs].values,
        Test_set[inputs].values,
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
