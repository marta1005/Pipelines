import pandas as pd
import surrogate_factory as sf


@sf.node
def batch_extract(workflow, input_table=None):
    """Load the fatigue dataset from Excel and filter to one SSE element."""
    metadata = workflow.get_step()
    path = metadata['path']
    sse_filter = metadata.get('sse_filter', None)

    dataset = pd.read_excel(path)

    if sse_filter is not None:
        dataset = dataset[dataset['SSE'] == int(sse_filter)].reset_index(drop=True)
        print(f"Filtered to SSE={sse_filter}: {dataset.shape[0]} rows")

    return dataset


@sf.node
def batch_transform(workflow, parsed_batches):
    """Strip leading/trailing whitespace from Type_segment values."""
    df = parsed_batches.copy()
    df['Type_segment'] = df['Type_segment'].str.strip()
    return df


@sf.node
def batch_load(workflow, transformed_batches, input_table=None):
    return transformed_batches


@sf.node
def data_visualization(workflow, input_table):
    try:
        from surrogate_factory.catalog.visualization.profiling import profiling_report
        profiling_report(input_table, "UCFatigue Raw Data")
    except Exception:
        print("Profiling not available.")
