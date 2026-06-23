import mlflow
import surrogate_factory as sf
import numpy.linalg as lin
import sklearn.metrics as skm
import mlflow
import numpy as np
import pandas as pd
from functools import partial
from cytoolz.curried import valfilter, complement
from cytoolz.curried.operator import is_



@sf.node
def calculate_metrics(workflow, Test_set, model_output):
    """_summary_
    Compute metrics, based in the models requirements defined.

    Args:
        flow_variables (_type_): flow_variables
        input_table (_type_): output table defined in the prediction.py
    """

    ## Example with some potential metrics:
       
    
    mlpipeline_metrics = {}
    for i, mlpipeline in enumerate(workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models'])):
        outputs = mlpipeline['outputs']

        y_pred = Test_set[outputs].copy()
        y_test = model_output[outputs].copy()
        
        categorical_features = []
        if (Test_set[outputs].dtypes == object).any() or (Test_set[outputs].dtypes == int).any():
                categorical_features = Test_set.select_dtypes([object, int]).columns.tolist()

        numeric_features = []
        if (Test_set[outputs].dtypes == float).any():
                numeric_features = Test_set[outputs].select_dtypes([float]).columns.tolist()
        
        
        compute_metrics = workflow.catalog.get_method("compute_metrics")
        compute_confusion_matrix = workflow.catalog.get_method("compute_confusion_matrix")
        
        metrics = {}
        for out in outputs:
            print("Computing ", out)
            if out in categorical_features:
                metrics[out] = compute_confusion_matrix(y_pred[[out]], y_test[[out]])
            elif out in numeric_features:
                metrics[out] = compute_metrics(y_pred[[out]], y_test[[out]])
            metrics_filtered = {key: value for key, value in metrics[out].items() if value is not None}
            log(mlpipeline['run_id'], metrics_filtered )
            mlpipeline_metrics[out] = metrics_filtered
    workflow.metadata.update_step_data({"scores":mlpipeline_metrics})
    return mlpipeline_metrics


def log(run_id, metrics):

        ## to be adapted to tracker solution
        ## example using mlflow as default tracker
        mlflow.start_run(run_id=run_id)
        mlflow.log_metrics(metrics)
        mlflow.end_run()