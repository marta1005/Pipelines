import pandas as pd
import joblib
import surrogate_factory as sf
from pathlib import Path


@sf.node
def train(workflow, Train_set, Val_set):
    """Train all algorithms listed in Model_Selection and save each model."""
    job_name = workflow.config['job_name']
    artifacts = Path(workflow.config['artifacts.folder'])

    ms = workflow.metadata.get_step_data(['metadata', 'Model_Selection'])
    inputs = ms['inputs']
    outputs = ms['outputs']
    algorithms = ms['algorithms']

    preprocess_file = workflow.metadata.get_step_data(
        ['metadata', 'Feature_Selection', 'Preprocess', 'file']
    )
    scaler = joblib.load(preprocess_file)

    Train_X = Train_set[inputs]
    Train_Y = Train_set[outputs]

    try:
        X_scaled = pd.DataFrame(
            scaler.transform(Train_X).toarray(),
            columns=scaler.get_feature_names_out()
        )
    except AttributeError:
        X_scaled = pd.DataFrame(
            scaler.transform(Train_X),
            columns=scaler.get_feature_names_out()
        )

    models_info = []

    for alg in algorithms:
        label = alg['label']
        settings = dict(alg['settings'])

        # YAML loads lists; MLPRegressor needs hidden_layer_sizes as tuple
        if 'hidden_layer_sizes' in settings and isinstance(settings['hidden_layer_sizes'], list):
            settings['hidden_layer_sizes'] = tuple(settings['hidden_layer_sizes'])

        # One tracker run per algorithm so metrics are kept separate in MLflow
        tracker_info = {'run_name': label, 'model': alg['name'], **settings}
        workflow.tracker.start_tracker(workflow, tracker_info)

        model_cls = workflow.catalog.get_method(alg['name'])
        model = model_cls(**settings)
        model.fit(X_scaled, Train_Y)

        # Log step-wise training curves to MLflow (no-op if default tracker)
        try:
            import mlflow as _mlflow
            if _mlflow.active_run():
                if hasattr(model, 'loss_curve_'):          # MLP
                    for step, loss in enumerate(model.loss_curve_):
                        _mlflow.log_metric('train_loss', loss, step=step)
                    val_scores = getattr(model, 'validation_scores_', None)
                    if val_scores is not None:
                        for step, score in enumerate(val_scores):
                            _mlflow.log_metric('val_score', score, step=step)
                    _mlflow.log_metric('n_iter', model.n_iter_)
                elif hasattr(model, 'estimators_'):        # MultiOutputRegressor (GB)
                    for col, est in zip(outputs, model.estimators_):
                        for step, score in enumerate(est.train_score_):
                            _mlflow.log_metric(f'train_loss_{col}', score, step=step)
        except ImportError:
            pass

        model_file = str(artifacts / f"model_{job_name}_{label}.modl")
        joblib.dump(model, model_file)

        info = {
            'label': label,
            'name': alg['name'],
            'inputs': inputs,
            'outputs': outputs,
            'file': model_file,
        }
        models_info.append(info)

        if hasattr(model, 'n_iter_'):                          # MLP
            n_iter = model.n_iter_
        elif hasattr(model, 'estimators_'):                    # MultiOutputRegressor
            inner = model.estimators_[0]
            n_iter = f"{inner.n_estimators} trees × {len(model.estimators_)} outputs"
        else:
            n_iter = getattr(model, 'n_estimators', '?')
        print(f"  [{label}] trained → {Path(model_file).name}  ({n_iter})")

        workflow.tracker.stop_tracker(workflow, 'FINISHED')

    workflow.metadata.update_step_data({'Models': models_info})

    return models_info
