
import numpy as np
import pandas as pd
import surrogate_factory as sf



@sf.node
def add_features(workflow, input_table)-> pd.DataFrame:
    """_summary_
    Add/Remove features according to Excel file. 
    This features will be used during training the model and stored with the model as a transformer
    for Production as any other preprocessor/postprocessor.
    Test Set doesn't contain these features/targets.
    Check with Deployment requirements
    Args:
        flow_variables (_type_): flow_variables
        input_table (_type_): Dataset

    Returns:
        pd.DataFrame: DataSet with new Features
    """
    

    return input_table
