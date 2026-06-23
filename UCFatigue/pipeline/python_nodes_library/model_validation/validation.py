import surrogate_factory as sf


@sf.node
def validate(workflow, metrics):
    """Compare computed metrics against requirements defined in SF_1."""
    accuracy_requirements = workflow.metadata.get_step_data(
        ['metadata', 'Requirements', 'accuracy']
    )

    for i, criteria in enumerate(accuracy_requirements):
        metric_name = criteria['metric']
        metric_output = criteria['output']
        max_value = criteria['value']

        score = metrics.get(metric_output, {}).get(metric_name)
        passed = str(score < max_value) if score is not None else "N/A"

        result = {
            "metric": metric_name,
            "output": metric_output,
            "target": max_value,
            "score": score,
            "passed": passed,
        }
        print(f"{metric_output} [{metric_name}]: {score:.4f} (target < {max_value}) → {'✓' if passed == 'True' else '✗'}")
        workflow.metadata.update_step_data(
            result, path=['metadata', 'Model_Validation', 'Validation', i]
        )
