import pandas as pd
import joblib
import surrogate_factory as sf


@sf.node
def predict(workflow, input_table) -> pd.DataFrame:
    """Apply preprocessor + each trained model. Returns columns <Label>__<output>."""
    inputs = workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'inputs'])
    outputs = workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'outputs'])
    preprocess_file = workflow.metadata.get_step_data(
        ['metadata', 'Feature_Selection', 'Preprocess', 'file']
    )
    models_info = workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models'])

    scaler = joblib.load(preprocess_file)
    X = input_table[inputs]
    try:
        X_scaled = pd.DataFrame(
            scaler.transform(X).toarray(),
            columns=scaler.get_feature_names_out()
        )
    except AttributeError:
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=scaler.get_feature_names_out()
        )

    result = {}
    for info in models_info:
        model = joblib.load(info['file'])
        y_pred = model.predict(X_scaled)
        for i, col in enumerate(outputs):
            result[f"{info['label']}__{col}"] = y_pred[:, i]

    return pd.DataFrame(result, index=input_table.index)
