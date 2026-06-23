import pickle
import numpy as np
import time
from pathlib import Path
import pandas as pd
import surrogate_factory as sf
import sympy as sp
from cytoolz import compose
from tensorflow import keras



##TODO: SF Core implemented models predict method.

import joblib

@sf.node
def predict_wrapper(workflow, input_table, outputs_cols=None) -> pd.DataFrame:

    import onnxruntime as rt
    result = pd.DataFrame()


    ## example using each model separately. onnxs can be merged into one single file to avoid custom inference engine.
    
    for wrapper_def in workflow.metadata.get_step_data(['metadata', 'Model_Deployment', 'models']):
        model_path = wrapper_def['onnx_file']
        output = wrapper_def['outputs']
        sess = rt.InferenceSession(model_path)
    
        inputs =  [input.name for input in sess.get_inputs()]
        output_names =  [outputs.name for outputs in sess.get_outputs()]
        
        onnx_type_to_np_dtype = {
                'tensor(float)': np.float32,
                'tensor(string)': object,
                'tensor(int64)': np.int64,
                'tensor(double)': np.float64,
            }
    
        input_data = {}
        for input_spec in sess.get_inputs():
            name = input_spec.name
            onnx_type = input_spec.type
    
            if name not in input_table:
                raise ValueError(f"Sample data not found for model input: {name}")
    
            dtype = onnx_type_to_np_dtype.get(onnx_type)
            if dtype is None:
                raise RuntimeError(f"Unsupported ONNX type '{onnx_type}' for input '{name}'")
    
            input_data[name] = np.array(input_table[name].values.reshape((-1,1)), dtype=dtype)
            # print(f"  - Input '{name}': created tensor with dtype {input_data[name].dtype}")
    
    
        result = pd.concat([result, pd.DataFrame(sess.run(None, input_data)[0], columns = output)], axis=1)


    return result



@sf.node
def predict(workflow, input_table, outputs_cols=None) -> pd.DataFrame:
    """This function will execute the preprocessor, model, postprocessor to compute the output

    Args:
        flow_variables (_type_): _description_
        input_table (_type_): _description_
        outputs_cols (_type_, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: a Dataframe containing each output in rows. [*inputs, y_pred, y_test, output_name]
    """


    predictions = {}
    for i, algorithm in enumerate(workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'algorithms'])):
        
        inputs = algorithm['inputs']
        outputs = algorithm['outputs']

        if outputs_cols != None:
            outputs = outputs_cols

        df = input_table[inputs]    
        
        
        output_name = outputs[0]
        
        ## apply preprocess

        for preprocess in workflow.metadata.get_step_data(['metadata','Feature_Selection','Preprocess','params' ]):
            if preprocess['outputs'] == algorithm['outputs']:
                scaler_path = preprocess['file']
        
        import joblib
        scaler = joblib.load(scaler_path)
        try:
            input_df_scaled = pd.DataFrame(scaler.transform(df).toarray(), columns=scaler.get_feature_names_out())
        except:
            input_df_scaled = pd.DataFrame(scaler.transform(df), columns=scaler.get_feature_names_out())

        ## model
        ### SF core contains a set of models available. You can use them or use your own model if the integration doesn't fit your need.
        for mlmodel in workflow.metadata.get_step_data(['metadata','Model_Training','Models']):
            if mlmodel['outputs'] == algorithm['outputs']:
                model_filename = mlmodel['file']
        model = joblib.load(model_filename)
        
        predictions[output_name] = model.predict(input_df_scaled)
        
            
    return pd.DataFrame(predictions)
        
        