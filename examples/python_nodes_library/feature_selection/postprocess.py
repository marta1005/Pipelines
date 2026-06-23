import numpy as np
import pandas as pd
import surrogate_factory as sf

import pickle
import os
from pathlib import Path
import joblib


@sf.node
def postprocessor(workflow, input_table):
    """_summary_
    Build model postprocess. Defeaturing
    postprocessors have to be saved in the models data folder and path to the file added into the flow_variables as any other relevant information to be able to load the postprocess in the following steps
    Args:
        flow_variables (_type_): flow_variables
        input_table (_type_): dataset for training
    """
    
    
    try:
        if len(workflow.get_step()['params']):
            postprocessor_fit = {}


            for i, postprocess_def in enumerate(workflow.get_step()):
            
                normalizer_transformer = workflow.catalog.get_method("normalizer_transformer")
                
                postprocessor_fit[i] = normalizer_transformer(input_table,postprocess_def )
                                                
                                    
                models_path = Path(workflow.config["artifacts.folder"])


                models_path.mkdir(exist_ok=True)

                file = str(models_path / f"postprocessor_{workflow.config['job_name']}_{i}.pkl")

                joblib.dump(postprocessor_fit[i], file)
                print(workflow.metadata.get_step_data( path = ['metadata', 'Feature_Selection', 'Postprocess','params',i]))
                
                workflow.metadata.update_step_data(data= {"file":file}, path = ['metadata', 'Feature_Selection', 'Postprocess','params',i])
    except:
        print("No postprocess method informed in params.") 
        workflow.metadata.update_step_data(data= {'Postprocess':None}, path = ['metadata', 'Feature_Selection'])



    
