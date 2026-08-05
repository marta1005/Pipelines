import pandas as pd
import surrogate_factory as sf


@sf.node
def batch_extract(workflow, input_table=None):
    """Load the airfoils dataset from CSV."""
    metadata = workflow.get_step()
    path = metadata['path']
    sep  = metadata.get('sep', ',')

    dataset = pd.read_csv(path, sep=sep)
    print(f"Loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    return dataset


@sf.node
def batch_transform(workflow, parsed_batches):
    """No transforms required — all columns are numeric."""
    return parsed_batches.copy()


@sf.node
def batch_load(workflow, transformed_batches, input_table=None):
    return transformed_batches
