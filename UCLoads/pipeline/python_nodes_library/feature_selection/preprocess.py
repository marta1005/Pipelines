import joblib
import surrogate_factory as sf
from pathlib import Path
from surrogate_factory.catalog.feature_selection.normalization import normalizer_transformer


@sf.node
def preprocessor(workflow, input_table):
    """Fit a StandardScaler for all numerical inputs and save it."""
    preprocess_cfg = workflow.metadata.get_step_data()
    add_remove_cfg = workflow.metadata.get_step_data(
        ['metadata', 'Feature_Selection', 'Add_Remove_Features']
    )

    params = dict(preprocess_cfg)
    params['outputs'] = add_remove_cfg['outputs']

    transformer = normalizer_transformer(input_table, params)

    artifacts = Path(workflow.config['artifacts.folder'])
    artifacts.mkdir(parents=True, exist_ok=True)
    file = str(artifacts / f"preprocess_{workflow.config['job_name']}.pkl")
    joblib.dump(transformer, file)

    workflow.metadata.update_step_data({'file': file})
    print(f"Saved preprocessor → {Path(file).name}")
    print(f"Feature names out: {list(transformer.get_feature_names_out())}")
