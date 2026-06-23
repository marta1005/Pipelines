import sys
import os.path

import pandas as pd
import numpy as np

import surrogate_factory as sf


@sf.node
def filter_outliers(workflow, input_table) -> pd.DataFrame:
    """_summary_
    Manage outliers. Develop a function to manage outliers
    Args:
        flow_variables (_type_): flow_variables
        input_table (_type_): Data cleaned without NaN values

    Returns:
        pd.DataFrame: data filtered from outliers
    """

    ## example of function that applies min_val-max_val range per output.
    
    #min_val = flow_variables...
    #max_val = flow_variables...
    #inputs_filtered = input_table[input_table[out].between(min_val, max_val)]
    # return inputs_filtered

    df = input_table.copy()
    input_outliers = workflow.metadata.get_step_data().get('input', None)
    output_outliers = workflow.metadata.get_step_data().get('output', None)
    print(input_outliers, output_outliers)
    if input_outliers is not None:
        for v in input_outliers:
            
            #input_table[v['name']].fillna(v['filling_value'], inplace=True)
            if "range" in v.keys():
                df = df[df[v['name']].between(v['range']['min'],v['range']['max'])]
            elif "value" in v.keys():
                df = df[df[v['name']]==v['value']]
            elif "options" in v.keys():
                df = df[df[v['name']].isin(v['options'])]
            print(v, df.shape)


    if output_outliers is not None:
        for v in output_outliers:
            #input_table[v['name']].fillna(v['filling_value'], inplace=True)
            if "range" in v.keys():
                df = df[df[v['name']].between(v['range']['min'],v['range']['max'])]
            elif "value" in v.keys():
                df = df[df[v['name']]==v['value']]
            elif "options" in v.keys():
                df = df[df[v['name']].isin(v['options'])]
            # df = df.iloc[indices]
            print(v, df.shape)

  
    
    return df#.iloc[indices]
