import pandas as pd
import surrogate_factory as sf


@sf.node
def add_features(workflow, input_table) -> pd.DataFrame:
    """Select configured inputs + outputs and add SSE as an integer feature.

    The SSE column (FEM element number) is retained as a numeric feature so that
    the model can be extended to multi-element training in the future.
    """
    step_data = workflow.metadata.get_step_data()
    inputs = step_data.get('inputs', [])
    outputs = step_data.get('outputs', [])
    keep_cols = inputs + [c for c in outputs if c not in inputs]

    df = input_table[[c for c in keep_cols if c in input_table.columns]].copy()

    # Ensure SSE is integer so the model treats it as a numeric feature
    if 'SSE' in df.columns:
        df['SSE'] = df['SSE'].astype(int)

    return df
