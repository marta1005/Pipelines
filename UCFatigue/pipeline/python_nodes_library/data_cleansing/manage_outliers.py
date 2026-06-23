import pandas as pd
import surrogate_factory as sf


@sf.node
def filter_outliers(workflow, input_table) -> pd.DataFrame:
    """Filter rows by range, exact value, or allowed options, per metadata config."""
    df = input_table.copy()
    step_data = workflow.metadata.get_step_data()
    input_filters = step_data.get('input', None)
    output_filters = step_data.get('output', None)

    def apply_filters(df, filters):
        for v in filters:
            col = v['name']
            if col not in df.columns:
                continue
            if 'range' in v:
                df = df[df[col].between(v['range']['min'], v['range']['max'])]
            elif 'value' in v:
                df = df[df[col] == v['value']]
            elif 'options' in v:
                df = df[df[col].isin(v['options'])]
            print(f"After filtering '{col}': {df.shape[0]} rows")
        return df

    if input_filters:
        df = apply_filters(df, input_filters)
    if output_filters:
        df = apply_filters(df, output_filters)

    return df.reset_index(drop=True)
