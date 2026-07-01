import joblib
import surrogate_factory as sf
from pathlib import Path
from sklearn.pipeline import Pipeline


@sf.node
def model_deployment(workflow):
    """Save a sklearn Pipeline (preprocessor + model) for every trained model."""
    preprocess_file = workflow.metadata.get_step_data(
        ['metadata', 'Feature_Selection', 'Preprocess', 'file']
    )
    models_info = workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models'])

    scaler = joblib.load(preprocess_file)
    artifacts = Path(workflow.config['artifacts.folder'])
    job_name = workflow.config['job_name']

    deployment_info = []
    for info in models_info:
        model = joblib.load(info['file'])
        sklearn_pipeline = Pipeline([('preprocessor', scaler), ('model', model)])
        pipeline_file = str(artifacts / f"pipeline_{job_name}_{info['label']}.pkl")
        joblib.dump(sklearn_pipeline, pipeline_file)
        deployment_info.append({
            'label': info['label'],
            'pipeline_file': pipeline_file,
            'outputs': info['outputs'],
        })
        print(f"Saved sklearn pipeline ({info['label']}) → {Path(pipeline_file).name}")

    workflow.metadata.update_step_data({'pipelines': deployment_info})
