import shutil
from pathlib import Path

import surrogate_factory as sf


def _storage_dir(workflow):
    """
    Resolve the configured storage path.

    A relative path is resolved against data.folder so the store travels with
    the use case; an absolute path (or a mounted share) is used as given.
    """
    store = workflow.metadata.get_step_data(
        ['metadata', 'Model_Storage', 'Store']
    ) or {}
    raw = store.get('storage_path') or 'model_store'
    p = Path(raw)
    if not p.is_absolute():
        p = Path(workflow.config['data.folder']) / p
    p.mkdir(parents=True, exist_ok=True)
    return p


@sf.node
def upload_model(workflow):
    """Copy the trained models and the fitted preprocessor into the store."""
    dest = _storage_dir(workflow)

    models = workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models']) or []
    preprocess = workflow.metadata.get_step_data(
        ['metadata', 'Feature_Selection', 'Preprocess']
    ) or {}

    stored = []
    for info in models:
        src = Path(info['file'])
        if not src.exists():
            print(f"  MISSING model for {info['label']}: {src}")
            continue
        shutil.copy2(src, dest / src.name)
        stored.append({'label': info['label'], 'file': str(dest / src.name)})
        print(f"  stored model    {info['label']:<20} -> {src.name}")

    pre = preprocess.get('file')
    if pre and Path(pre).exists():
        shutil.copy2(pre, dest / Path(pre).name)
        stored.append({'label': 'preprocessor', 'file': str(dest / Path(pre).name)})
        print(f"  stored preproc  {'':<20} -> {Path(pre).name}")

    # Explicit path: these node names are not in the framework's process mapping,
    # so current_step would be empty and the data would land at the metadata root.
    workflow.metadata.update_step_data(
        {'storage_path': str(dest), 'models': stored},
        ['metadata', 'Model_Storage', 'Store'],
    )
    print(f"\n  {len(stored)} artifact(s) in {dest}")
    return stored
