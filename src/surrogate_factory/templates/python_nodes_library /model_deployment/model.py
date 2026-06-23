import os
import logging
import sys
from pandas import DataFrame
import random
import warnings
import numpy as np
import logging

from pathlib import Path
import surrogate_factory as sf

import mlflow
import shutil
import sklearn




@sf.node
def model_deployment(workflow):
    """Prepare model for deployment.

    Args:
        flow_variables (_type_): _description_
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    ### not implemented in this example
    models_onnx = []
    if (workflow.get_step()['method']=="ONNX"):
        # from surrogate_factory.catalog.deploy import onnx_converter
        convert_sklearn_to_onnx = workflow.catalog.get_method("convert_sklearn_to_onnx")
        import joblib

        
        for i, algorithm in enumerate(workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'algorithms'])):

            for preprocess in workflow.metadata.get_step_data(['metadata','Feature_Selection','Preprocess','params' ]):
                if preprocess['outputs'] == algorithm['outputs']:
                    scaler_path = preprocess['file']
                    print(scaler_path)

            for mlmodel in workflow.metadata.get_step_data(['metadata','Model_Training','Models']):
                if mlmodel['outputs'] == algorithm['outputs']:
                    model_filename = mlmodel['file']

            models_path = Path(workflow.config["artifacts.folder"])    
            # if workflow.metadata.get_step_data()['method'] == 'ONNX':
            file_onnx = models_path/f"{Path(model_filename).stem}.onnx"
            convert_sklearn_to_onnx(scaler_path, model_filename, file_onnx)
                
            workflow.metadata.update_step_data({
                                                'preprocess_file': str(scaler_path),
                                                'mlmodel_file':str(model_filename),
                                                'onnx_file':str(file_onnx),
                                                'outputs': algorithm['outputs']
                                                },
                                                path=['metadata','Model_Deployment', 'models',i]
                                                )



        if len(models_onnx)>1:
            merge_onnx_models = workflow.catalog.get_method("merge_onnx_models")
            wrapper_path = models_path/"model_wrapper.onnx"
            merge_onnx_models(models_onnx,wrapper_path )
            workflow.metadata.update_step_data({
                                                'wrapper_path': str(wrapper_path),
                                                },
                                                
                                                )
