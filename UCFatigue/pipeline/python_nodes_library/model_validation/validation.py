import surrogate_factory as sf


@sf.node
def validate(workflow, metrics):
    """Check best-model score per output against requirements from SF_1."""
    accuracy_requirements = workflow.metadata.get_step_data(
        ['metadata', 'Requirements', 'accuracy']
    )

    # metrics structure: {label: {output: {metric_name: value}}}
    labels = list(metrics.keys())

    print(f"\n  {'Output':<28}{'Metric':<16}{'Target':<12}", end="")
    for label in labels:
        print(f"{label:>14}", end="")
    print()
    print("  " + "-" * (56 + 14 * len(labels)))

    all_results = []
    for criteria in accuracy_requirements:
        metric_name = criteria['metric']
        output = criteria['output']
        target = criteria['value']

        row = f"  {output:<28}{metric_name:<16}{target:<12.4f}"
        result_entry = {'output': output, 'metric': metric_name, 'target': target, 'models': {}}

        for label in labels:
            score = (metrics.get(label, {}).get(output, {}).get(metric_name))
            if score is not None:
                passed = score < target
                row += f"{'✓' if passed else '✗'} {score:>10.4f}"
                result_entry['models'][label] = {'score': score, 'passed': passed}
            else:
                row += f"{'?':>14}"
                result_entry['models'][label] = {'score': None, 'passed': None}

        print(row)
        all_results.append(result_entry)

    workflow.metadata.update_step_data({'validation_results': all_results})
