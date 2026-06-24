
import pathlib
import os
import sys
import importlib
import mlflow


import mlflow
from mlflow.tracking import MlflowClient


from typing import Dict, Union, Any
import pandas as pd

from toolz import curry, memoize
run_once = memoize(key=lambda args, kwargs: None)

tracker_name = "mlflow"
plugin = True

@run_once
def setup_mlflow(flow_variables: Dict[str, Any]):
    """ Set up mlflow environment """
    # mlflow.set_tracking_uri("file:///projects/gr-st_A350_airbus_fuselage_machinelearning/SurrogateFactory/mlruns")
    tracker_options = flow_variables['tracker']['options']

    # if "tracking_uri" in  tracker_options.keys():
    #     tracking_uri = tracker_options['tracking_uri']
    # else:
    #     tracking_uri = "file:///projects/gr-st_A350_airbus_fuselage_machinelearning/SurrogateFactory/mlruns"

    try:
        print("Setting tracking uri:", tracker_options['tracking_uri'])
        mlflow.set_tracking_uri(tracker_options['tracking_uri'])
    except:
        raise Exception("Tracking uri missing or not available")
    
    # if "artifactory_repo" in tracker_options:
    artifactory_repo = tracker_options.get("artifactory_repo", None) #R-E32B_HiveOps_generic_L"
    repo_key = tracker_options.get("repo_key", None)

    if "apikey_path" in tracker_options.keys():
        
        if tracker_options['apikey_path'] != '':
            apikey_path = pathlib.Path(tracker_options['apikey_path'])
            if apikey_path != '.':
                with apikey_path.open('r') as f:
                    apikey = f.read().strip()
        else:
            apikey = ''

        os.environ["MLFLOW_ARTIFACTORY_ENDPOINT_URL"] = artifactory_repo
        os.environ["MLFLOW_ARTIFACTORY_KEY"] = apikey  ## TOKEN IMPLEMENTATION
        os.environ["MLFLOW_ARTIFACTORY_REPO"] = repo_key

        if sys.platform.startswith('linux'):
            ca_path = pathlib.Path(tracker_options['pem_cert_path'])
            if ca_path.exists():
                os.environ["REQUESTS_CA_BUNDLE"] = str(ca_path)
    pass

def _flatten_params(d, prefix=""):
    """Flatten nested dict to MLflow-compatible {str: str} params."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            items.update(_flatten_params(v, key))
        elif isinstance(v, (list, tuple)):
            items[key] = str(v)
        else:
            items[key] = str(v)
    return items


def start_tracker(workflow, algorithm):
    experiment_name = workflow.config['job_name']
    workflow.metadata.update_step_data({'Tracking': {'experiment_name': experiment_name}})

    # Create or reuse the experiment (artifactory location silently ignored locally)
    try:
        mlflow.create_experiment(experiment_name)
    except Exception:
        pass
    mlflow.set_experiment(experiment_name)

    run_name = algorithm.get('run_name', algorithm.get('label', 'run'))
    mlflow.start_run(run_name=run_name)

    run = mlflow.active_run()
    run_id = run.info.run_id
    print(f"MLflow run started: {run_name}  (id: {run_id})")

    workflow.metadata.update_step_data({
        'Tracking': {'eval': {run_name: {'run_id': run_id}}}
    })

    # Log flattened params (MLflow requires str values, no nested dicts)
    flat = _flatten_params({k: v for k, v in algorithm.items() if k != 'run_name'})
    if flat:
        mlflow.log_params(flat)

    mlflow.autolog(log_models=True, silent=True)

    return run_id


def stop_tracker(workflow, status):
    # Map surrogate_factory status strings to valid MLflow terminal states
    status_map = {
        'RUNNING':  'FINISHED',
        'FINISHED': 'FINISHED',
        'FAILED':   'FAILED',
        'KILLED':   'KILLED',
    }
    mlflow.end_run(status=status_map.get(status, 'FINISHED'))


def log(run_id, metrics):
    mlflow.start_run(run_id=run_id)
    mlflow.log_metrics(metrics)
    mlflow.end_run()