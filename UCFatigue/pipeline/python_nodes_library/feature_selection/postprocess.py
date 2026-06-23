import surrogate_factory as sf


@sf.node
def postprocessor(workflow, input_table):
    """Postprocessing step — not required for this use case."""
    print("No postprocess method configured.")
