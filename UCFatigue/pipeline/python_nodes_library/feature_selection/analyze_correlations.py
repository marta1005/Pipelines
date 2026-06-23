import surrogate_factory as sf


@sf.node
def explore_data_analysis(workflow, input_table):
    """Generate a profiling report for the training set."""
    try:
        from surrogate_factory.catalog.visualization.profiling import profiling_report
        profiling_report(input_table, "UCFatigue Feature Selection EDA")
    except Exception:
        print("Profiling not available.")
