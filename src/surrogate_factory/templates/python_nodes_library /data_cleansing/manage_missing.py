import pandas as pd
import surrogate_factory as sf


@sf.node
def replace_missing_values(workflow, input_table: pd.DataFrame)-> pd.DataFrame:
    """_summary_
    According to flow_variables, this process manages NaN values.
    
    Args:
        flow_variables (sf.FlowVariables): flow_variables
        input_table (pd.DataFrame): raw data

    Returns:
        pd.DataFrame: Data cleaned
    """
    ## this is an example how to fill each column with the value "filling_value" in the excel file


    outputs = workflow.metadata.get_step_data().get('output', None)
    if outputs is not None:
        for v in outputs:
            input_table[v['name']].fillna(v['filling_value'], inplace=True) 
            
        return input_table
    else:

        print("No change applied")
        return input_table
