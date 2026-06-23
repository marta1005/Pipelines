import numpy as np
import pandas as pd
import surrogate_factory as sf

from pathlib import Path
import joblib

@sf.node
def preprocessor(workflow, input_table):
    """_summary_
    Fit preprocess according to the excel file.
    preprocessors have to be saved in the models data folder and path to the file added into the flow_variables as any other relevant information to be able to load the preprocess in the following steps
    Args:
        flow_variables (_type_): _description_
        input_table (_type_): _description_
    """

    ## Example of function 

    try:
        if len(workflow.get_step()):
            preprocess_fit = {}


            for i, preprocess_def in enumerate(workflow.get_step()['params']):

                if preprocess_def['method'] == "normalizer_transformer":
            
                    normalizer_transformer = workflow.catalog.get_method("normalizer_transformer")
                    
                    preprocess_fit[i] = normalizer_transformer(input_table, preprocess_def)
                                                    
                                        
                    models_path = Path(workflow.config["artifacts.folder"])
        
        
                    models_path.mkdir(exist_ok=True)
        
                    file = str(models_path / f"preprocess_{workflow.config['job_name']}_{i}.pkl")
        
                    joblib.dump(preprocess_fit[i], file)
                    print(workflow.metadata.get_step_data( path = ['metadata', 'Feature_Selection', 'Preprocess','params',i]))
                    
                    workflow.metadata.update_step_data(data= {"file":file}, path = ['metadata', 'Feature_Selection', 'Preprocess','params',i])
    except:
        print("No preprocess method informed in params.") 
        workflow.metadata.update_step_data(data= {'Preprocess':None}, path = ['metadata', 'Feature_Selection'])