import numpy as np
import pandas as pd
import surrogate_factory as sf

import pickle
import os

@sf.node
def data_split(workflow, input_table):
    """_summary_
    Function to split data in the 3 datasets for training, test and validation
    flow_variables should be updated accordingly
    Args:
        flow_variables (_type_): flow_variables of the pipeline 
        input_table (_type_): Dataset to split

    Returns:
        trainX: pandas DataFrame
        testX: pandas DataFrame
        valX: pandas DataFrame
        trainY: pandas DataFrame
        testY: pandas DataFrame
        valY: pandas DataFrame
    """
    from sklearn.model_selection import train_test_split

    ## READ PARAMS FROM METADATA
    params = workflow.metadata.get_step_data()
    train_perc = params["percentages"].get("train", 0.8)
    test_perc = params["percentages"].get("test", 0.2)
    validation_perc = params["percentages"].get("validation", 0.2)

    print(train_perc,test_perc,  validation_perc)

    Train_set_, Val_set = train_test_split(
                                                input_table,
                                                test_size=validation_perc, random_state=42, shuffle=True
                                                )
    Train_set, Test_set = train_test_split(
                                                Train_set_,
                                                test_size=test_perc, random_state=42, shuffle=True
                                                )
   

    ## UPDATES METADATA
    workflow.metadata.update_step_data({'size':{
                                        '       train': Train_set.shape[0],
                                                'test': Test_set.shape[0],
                                                'validation': Val_set.shape[0]   }
                                                }
                                        )
    
    

    return Train_set,Test_set,Val_set