import surrogate_factory as sf


@sf.node
def validate(workflow, metrics):
    """This function check metrics of the models with requirements
    

    Args:
        workflow (_type_): _description_
        metrics (_type_): _description_

    Returns:
        _type_: Flow_variables Model_Validation updated
    """
    #Example of requirements vs metrics comparison
    accuracy_requirements = workflow.metadata.get_step_data(['metadata','Requirements','accuracy'])


    for i, criteria in enumerate(accuracy_requirements):
            metric_name = criteria['metric']
            metric_output = criteria['output']
            max_value = criteria['value']
            

            criteria_evaluation = {"metric": criteria['metric'], "output":metric_output, "target":max_value, "score":metrics[metric_output][metric_name], "passed":False}
            
            try:
                criteria_evaluation["passed"] = str(metrics[metric_output][metric_name] < max_value)
                workflow.metadata.update_step_data(criteria_evaluation, path = ['metadata', 'Model_Validation', 'Validation', i])
                
            except:
                print(f"metric: {metric_name} not computed")
                workflow.metadata.update_step_data(criteria_evaluation, path = ['metadata', 'Model_Validation', 'Validation', i])
        
    