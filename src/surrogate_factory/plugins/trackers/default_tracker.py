# surrogate_factory/plugins/trackers/default_tracker.py

import os

class DefaultTracker:
    """
    A simple, default experiment tracker that prints logs to the console.
    This tracker is used when no other tracking service (like MLflow) is configured.
    It provides a basic level of logging for key events in the pipeline and mimics
    the structure of more advanced trackers like MLflow for compatibility.
    """

    def __init__(self, tracking_uri: str = None, experiment_name: str = "Default_Experiment", **kwargs):
        """
        Initializes the DefaultTracker.
        It accepts arguments to maintain compatibility with other trackers but only prints them.
        
        Args:
            tracking_uri (str, optional): The path to the tracking directory. Defaults to None.
            experiment_name (str, optional): The name of the experiment. Defaults to "Default_Experiment".
        """
        print("Initializing DefaultTracker: All logs will be printed to the console.")
        self.run_name = "default_run"
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        if self.tracking_uri:
            print(f"[CONFIG] Tracking URI set to: {self.tracking_uri}")
        
        self.set_experiment(self.experiment_name)

    def set_experiment(self, experiment_name: str):
        """
        Sets the experiment name for all subsequent runs.
        
        Args:
            experiment_name (str): The name of the experiment.
        """
        self.experiment_name = experiment_name
        print(f"[CONFIG] Experiment set to: '{self.experiment_name}'")

    def start_run(self, run_name: str = "default_run"):
        """
        Starts a new run. In this tracker, it just prints a message.
        
        Args:
            run_name (str): The name of the run to start.
        """
        self.run_name = run_name
        print(f"\n--- Starting Run: {self.run_name} ---")

    def log_param(self, key: str, value):
        """
        Logs a single parameter.
        
        Args:
            key (str): The name of the parameter.
            value: The value of the parameter.
        """
        print(f"[PARAM] {key}: {value}")

    def log_params(self, params: dict):
        """
        Logs a dictionary of parameters.
        
        Args:
            params (dict): A dictionary of parameters to log.
        """
        print("--- Logging Parameters ---")
        for key, value in params.items():
            self.log_param(key, value)
        print("--------------------------")

    def log_metric(self, key: str, value: float, step: int = None):
        """
        Logs a single metric.
        
        Args:
            key (str): The name of the metric.
            value (float): The value of the metric.
            step (int, optional): The step or epoch for the metric. Defaults to None.
        """
        step_info = f" (Step: {step})" if step is not None else ""
        print(f"[METRIC] {key}: {value}{step_info}")

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Logs a dictionary of metrics.
        
        Args:
            metrics (dict): A dictionary of metrics to log.
            step (int, optional): The step or epoch for the metrics. Defaults to None.
        """
        step_info = f" (Step: {step})" if step is not None else ""
        print(f"--- Logging Metrics{step_info} ---")
        for key, value in metrics.items():
            self.log_metric(key, value)
        print("-------------------------")

    def set_tag(self, key: str, value):
        """
        Sets a tag for the run.
        
        Args:
            key (str): The name of the tag.
            value: The value of the tag.
        """
        print(f"[TAG] {key}: {value}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Logs a local file as an artifact.
        
        Args:
            local_path (str): Path to the file to log.
            artifact_path (str, optional): Directory within the run's artifact URI to log to.
        """
        print(f"[ARTIFACT] Logging file from '{local_path}' to artifact path '{artifact_path or ''}'")

    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """
        Logs all the files in a local directory as artifacts.

        Args:
            local_dir (str): Path to the directory to log.
            artifact_path (str, optional): Directory within the run's artifact URI to log to.
        """
        print(f"[ARTIFACTS] Logging directory from '{local_dir}' to artifact path '{artifact_path or ''}'")

    def end_run(self):
        """
        Ends the current run. In this tracker, it just prints a message.
        """
        print(f"--- Ending Run: {self.run_name} ---\n")

def start_tracker(workflow, algorithm):
    """This function will use your tracker to train the model provided

    Args:
        flow_variables (_type_): _description_
        model (_type_): _description_
        train_set (_type_): _description_
        val_set (_type_, optional): _description_. Defaults to None.
    """
    experiment_name = workflow.config['job_name']
    workflow.metadata.update_step_data({'Tracking':{'experiment_name': experiment_name}})
    # workdir = Path(experiment_name)


    ## Set MLFlow tracker
    # try:
    #     mlflow.create_experiment(experiment_name, artifact_location="artifactory://sf-mlflow")
    # except Exception:
    #     pass

    # mlflow.set_experiment(experiment_name)

    # run_name = algorithm['output']
    # mlflow.start_run(run_name=run_name)

    # run = mlflow.active_run()
    # run_id = run.info.run_id
    # print("run id:",run.info.run_id)

    # workflow.metadata.update_step_data({'Tracking':{'eval':{run_name: {'run_id': run_id,
    #                                                                 'algorithm': algorithm}}}}) #run.info.run_id
    # # flow_variables['metadata', 'Model_Training', 'Tracking', 'eval', flow_variables['currentColumnName'], 'params'] = params


    # mlflow.log_params(algorithm)

    # mlflow.autolog()

    run_id = 0000

    return run_id

def stop_tracker(workflow, status):
    # mlflow.end_run(status=status)
    print("Ended Default Tracking")