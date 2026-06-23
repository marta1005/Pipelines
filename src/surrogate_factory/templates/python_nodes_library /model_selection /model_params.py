from IPython.display import display, HTML, JSON
import sys
import os.path

import json

from pathlib import Path
import json
import pandas as pd 
import surrogate_factory as sf

## TODO: Implement different hypertuning methods in the Core
# from surrogate_factory.workflow.tune import  Hypertuning_params



@sf.node
def define_model(workflow, input_table: pd.DataFrame): 
    job_name = workflow.config['job_name']
        
    params =  workflow.metadata.get_step_data()
    
    ## Example of Hypertuning execution with Optuna method from Catalog
    if "hypertuning" in params.keys():
        
        
        models = {}
    
        for i, method in enumerate(params['hypertuning']):
        
            if 'inputs' not in method.keys():
                inputs = workflow.metadata.get_step_data(['metadata','Feature_Selection','Preprocess','params',0,'input'])
            
            outputs = method['output']
    
            if isinstance(outputs, str):
                outputs = list(map(str.strip, outputs.strip('[]').split(',')))
    
            settings = method['settings']#{"algorithm":method['algorithm'], "settings":method['settings']}
          
            print("settings",settings)
            ## apply preprocess
    
            
            for preprocess in workflow.metadata.get_step_data(['metadata','Feature_Selection','Preprocess','params' ]):
                print(method['output'], preprocess['output'])
                if (preprocess['output'] == outputs) or (set(outputs).issubset(preprocess['output'])):
                    scaler_path = preprocess['file']
            Train_X = input_table[inputs].copy()
            Train_Y = input_table[outputs].copy()
            
            import joblib
            scaler = joblib.load(scaler_path)
            try:
                input_train_scaled = pd.DataFrame(scaler.transform(Train_X).toarray(), columns=scaler.get_feature_names_out())
                # input_val_scaled = pd.DataFrame(scaler.transform(Val_X).toarray(), columns=scaler.get_feature_names_out())
            except:
                input_train_scaled = pd.DataFrame(scaler.transform(Train_X), columns=scaler.get_feature_names_out())
                # input_val_scaled = pd.DataFrame(scaler.transform(Val_X), columns=scaler.get_feature_names_out())
            ## fit model
            ### SF core contains a set of models available. You can use them or use your own model if the integration doesn't fit your need.
    
            if len(method['output'])>1:
                output_name = "_".join(outputs)
            if isinstance(outputs, list): # and len(settings['output'])==0:
                output_name = outputs[0]
    
            
            print(output_name, method['output'])
           
            
            MODEL_NAME = settings['model']#"MLPRegressor"
            
            # 7. Get model class
            ## this method is using modelfactory model wrapper
            # model_class = modelfactory.get_model_wrapper({
            #                                                 "algorithm": MODEL_NAME,
            #                                             })._model_class
            
            # With the Catalog, model class can be retrieved like this. Update pipeline config to add models according Catalog Documentation
            model_class = workflow.catalog.get_method(MODEL_NAME)
        
            if method['method'] == "tune_model":
                
                tune_model = workflow.workflow.get_method("tune_model")
                # Run the tuning
                best_hyperparams = tune_model(
                    input_train_scaled, 
                    Train_Y, 
                    model_class=model_class,
                    search_space=settings['search_space'],#SEARCH_SPACE, 
                    static_settings=settings['static_params'],
                    n_trials=30 # More trials for this complex model
                )
                
    
        
                workflow.metadata.update_step_data({            'inputs': inputs,
                                                                'outputs':outputs,
                                                                'settings': best_hyperparams,
                                                                "algorithm":method['settings']['model']
                                                                },
                                                    path=['metadata','Model_Selection','best_params',i])
        
                
                models[i] = best_hyperparams
            else:
                print("Method {settings['method'} not available")
            # break
        return models 

    