import sys
import os.path

import surrogate_factory as sf
import pandas as pd


@sf.node
def launch_sim(workflow: sf.Workflow, input_table: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    This function will execute the Simulation tool with the Input_table provided to generate the Output results

    
    Args:
        workflow (sf.Workflow): Workflow
        input_table (pd.DataFrame): input data to use

    Returns:
        pd.DataFrame: optional
    """


    pass


