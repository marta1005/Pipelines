import numpy as np
import pandas as pd
import surrogate_factory as sf

@sf.node
def doe_size_check(flow_variables, input_table):
    """_summary_
    Example of DoE consistancy checks.
    This example checks DoE size specified in metadata is the same than data generated.

    Args:
        flow_variables (_type_): flow_variables
        input_table (_type_): DoE
    """
    doe_size = flow_variables['metadata', 'Data_acquisition_generation', 'Data_generation', 'Create_Design_of_Experiments', 'DoE', 'size']
    if len(input_table) == doe_size:
        print("\033[1;32m Sanity checks successfull.  \n")
    else:
        print("\033[1;31m Mismatches found. Please check.  \n")
        print(f" Expected DoE Size :{doe_size}")
        print(f" Got No of observations :{len(input_table)}")