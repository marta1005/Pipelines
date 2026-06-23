import pandas as pd
import numpy as np

from pathlib import Path
from toolz.curried import get, first, compose
from toolz.curried.operator import eq

import surrogate_factory as sf



@sf.node
def train(workflow: sf.Workflow, Train_set: pd.DataFrame, Val_set: pd.DataFrame, outputs_cols=None):
    """
    Function to train models. 
    It contains 3 steps:
      1.- Split data into TrainX. TrainY, (ValX and ValY)
      2.- Depending if it's a multioutput model or model per output, separate data accordingly
      3.- Load and apply preprocessor to TrainX and ValX
      4.- Get Model as defined in Model Selection step
      5.- Start Model tracker
      6.- Fit the model
      7.- Stop tracking and save all data. Model's files
      8.- Add in Flow_variables the path to the model
    Args:
        workflow (sf.workflow): our workflow class
        input_table_1 (pd.DataFrame): train data
        
    Requeriments:
        Use a Model Tracker (as MlFLow) to track your model
    Returns:
        a model fitted
        Models has to be saved and path to files added in the metadata accordingly.
        Models files will be used by next functions and notebooks in the pipeline
    """




    ## This is an example. 
    job_name = workflow.config['job_name']
    
    params =  workflow.metadata.get_step_data()
    workflow.metadata.update_step_data({"Tracking":{'experiment_name':job_name}})
    tracker = workflow.tracker

    models = {}

    for i, algorithm in enumerate(workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'algorithms'])):
        
        inputs = algorithm['inputs']
        outputs = algorithm['outputs']

        if outputs_cols != None:
            outputs = outputs_cols

        Train_X = Train_set[inputs]
        Train_Y = Train_set[outputs] 
        Val_X = Val_set[inputs]
        Val_Y = Val_set[outputs] 
        
        algorithm['run_name'] = outputs[0]

        run_id = tracker.start_tracker(workflow, algorithm)

        ## apply preprocess

        for preprocess in workflow.metadata.get_step_data(['metadata','Feature_Selection','Preprocess','params' ]):
            if preprocess['outputs'] == algorithm['outputs']:
                scaler_path = preprocess['file']
        
        import joblib
        scaler = joblib.load(scaler_path)
        try:
            input_train_scaled = pd.DataFrame(scaler.transform(Train_X).toarray(), columns=scaler.get_feature_names_out())
        except:
            input_train_scaled = pd.DataFrame(scaler.transform(Train_X), columns=scaler.get_feature_names_out())
        ## fit model
        ### SF core contains a set of models available. You can use them or use your own model if the integration doesn't fit your need.

        ## Note: Model selected in the following example is Sklearn base model. Model compile/build is not needed. 
        model = workflow.catalog.get_method(algorithm['algorithm'])(**algorithm['settings'])
        
        model.fit(input_train_scaled, Train_Y)

        models[algorithm['run_name']] = model

        ## Models has to be saved and path to files added in the metadata accordingly.
        ## Models files will be used by next functions and notebooks in the 
        

        models_path = Path(workflow.config["artifacts.folder"])
        model_file = models_path / f"model_{workflow.config['job_name']}_{algorithm['run_name']}_{run_id}.modl"
        joblib.dump(models[algorithm['run_name']], model_file )  
        

        workflow.metadata.update_step_data({               
                                                          'inputs':algorithm['inputs'], 
                                                          'outputs':algorithm['outputs'],
                                                          'file': str(model_file),
                                                          "run_id":run_id
                                                         },
                                            path=['metadata','Model_Training','Models',i])

        tracker.stop_tracker(workflow, "RUNNING")


    return models 