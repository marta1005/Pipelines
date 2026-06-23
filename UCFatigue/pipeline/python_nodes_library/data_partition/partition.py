import pandas as pd
import surrogate_factory as sf
from sklearn.model_selection import train_test_split


@sf.node
def data_split(workflow, input_table):
    """Split dataset into train, test and validation sets."""
    params = workflow.metadata.get_step_data()
    train_perc = params["percentages"].get("train", 0.8)
    test_perc = params["percentages"].get("test", 0.1)
    validation_perc = params["percentages"].get("validation", 0.1)

    Train_set_, Val_set = train_test_split(
        input_table, test_size=validation_perc, random_state=42, shuffle=True
    )
    Train_set, Test_set = train_test_split(
        Train_set_, test_size=test_perc, random_state=42, shuffle=True
    )

    workflow.metadata.update_step_data({
        'size': {
            'train': Train_set.shape[0],
            'test': Test_set.shape[0],
            'validation': Val_set.shape[0],
        }
    })

    print(f"Train: {Train_set.shape[0]} | Test: {Test_set.shape[0]} | Val: {Val_set.shape[0]}")
    return Train_set, Test_set, Val_set
