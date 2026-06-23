import sys
import os.path
import pandas as pd
import numpy as np

import surrogate_factory as sf
"""import the DoE library of your choice. 
SF contains one implentation by default
The requirements of the library is that the function used has to be declared
and parametrized in the excel file and serialized in the metadata
to be able to reproduce the list of values
Note that restrictions and dependencies has also be considered
in the declarations
"""
# from doe import DoE


@sf.node
def create_inputs(workflow) -> pd.DataFrame:
    """_summary_
    Build your DoE matrix of values based on the DoE definition.
    Normally in these point is where the sequence selected is executed with all inputs defined.
    it returns an array of values between 0-1
    Args:
        flow_variables (_type_): flow_variables

    Returns:
        pd.DataFrame: DoE dataframe
    """

    pass


@sf.node
def scale(workflow, input_table)-> pd.DataFrame:
    """_summary_
    it scales between bounds the doe previosly generated
    Args:
        flow_variables (_type_): flow_variables
        input_table (_type_): DoE dataframe to scale

    Returns:
        pd.DataFrame: dataframe scaled
    """



    pass


@sf.node
def extend(workflow, input_table)->pd.DataFrame:
    """_summary_
    Extends results if applicable
    Args:
        flow_variables (_type_): flow_variables
        input_table (_type_): DoE scaled

    Returns:
        pd.DataFrame: DataFrame with all variables computed
    """

    pass


@sf.node
def filter(workflow, input_table)-> pd.DataFrame:
    """_summary_
    Filter if applicable. Resolve any dependency and apply restrictions
    Args:
        flow_variables (_type_): flow_variables
        input_table (_type_): DoE after extend

    Returns:
        pd.DataFrame: Doe with all variables computed and scaled
    """

    pass
