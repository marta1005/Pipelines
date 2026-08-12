import json
import shutil
from pathlib import Path

import surrogate_factory as sf

from storage.model import _storage_dir


@sf.node
def save_pipeline(workflow):
    """
    Copy the deployable sklearn pipelines (.pkl) into the store and drop a
    manifest next to them, so the store is self-describing without the
    workflow metadata.
    """
    dest = _storage_dir(workflow)

    pipelines = workflow.metadata.get_step_data(
        ['metadata', 'Model_Deployment', 'pipelines']
    ) or []
    sel = workflow.metadata.get_step_data(['metadata', 'Model_Selection']) or {}

    saved = []
    for info in pipelines:
        src = Path(info['pipeline_file'])
        if not src.exists():
            print(f"  MISSING pipeline for {info['label']}: {src}")
            continue
        shutil.copy2(src, dest / src.name)
        saved.append({'label': info['label'],
                      'pipeline_file': str(dest / src.name),
                      'outputs': info.get('outputs', [])})
        print(f"  stored pipeline {info['label']:<20} -> {src.name}")

    manifest = {
        'job_name': workflow.config['job_name'],
        'inputs': sel.get('inputs', []),
        'outputs': sel.get('outputs', []),
        'pipelines': saved,
    }
    manifest_path = dest / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest        -> {manifest_path.name}")

    workflow.metadata.update_step_data(
        {'pipelines': saved, 'manifest': str(manifest_path)},
        ['metadata', 'Model_Storage', 'Store'],
    )
    return saved
