import pandas as pd
import surrogate_factory as sf


@sf.node
def drop_missing(workflow, input_table: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN values (all inputs and outputs are numerical)."""
    before = len(input_table)
    df = input_table.dropna().reset_index(drop=True)
    after = len(df)
    if before > after:
        print(f"Dropped {before - after} rows with NaN values ({before} → {after})")
    else:
        print(f"No missing values found — dataset unchanged ({after} rows)")
    return df


@sf.node
def report_statistics(workflow, input_table: pd.DataFrame):
    """Print basic statistics and return the table unchanged."""
    print(input_table.describe().to_string())
    return input_table
