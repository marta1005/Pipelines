
import surrogate_factory as sf
import pandas as pd


@sf.node
def explore_data_analysis(workflow, input_table: pd.DataFrame):
    """_summary_
    Build some plots for EDA.
    Report should be stored with the model during model storage.
    Use this report to select features with the proper justication
    Args:
        flow_variables (_type_): flow_variables
        input_table (_type_): pd.DataFrame containing the dataset
    """

    ## This example is using SF ydata_profiling integrated report
    
    # inputs = workflow.metadata.get_step_data()['inputs']
    # outputs = workflow.metadata.get_step_data()['outputs']

    title = f"Profiling Report {workflow.config['job_name']}"

    method = workflow.catalog.get_method("profiling_report")
    profile = method(input_table, title=title)
    
    # workflow.metadata.update_step_data({"eda_report": json.loads(profile.json)})