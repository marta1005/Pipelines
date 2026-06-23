import pandas as pd
import surrogate_factory as sf


@sf.node
def replace_missing_values(workflow, input_table: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values as configured in metadata.

    Supports filling_value as a number or 'median'.
    """
    outputs = workflow.metadata.get_step_data().get('output', None)
    if outputs is None:
        print("No missing-value config found, skipping.")
        return input_table

    df = input_table.copy()
    for v in outputs:
        col = v['name']
        fill = v['filling_value']
        if col not in df.columns:
            continue
        if fill == 'median':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(fill)

    return df
