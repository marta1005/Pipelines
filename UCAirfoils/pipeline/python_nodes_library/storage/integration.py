import json
from pathlib import Path

import numpy as np
import surrogate_factory as sf

from storage.model import _storage_dir


@sf.node
def model_test(workflow, Test_set=None):
    """
    Integration test on the *stored* artifacts: reload each pipeline straight
    from the store and predict, so a store that cannot be consumed downstream
    fails here rather than in production.
    """
    import joblib

    dest = _storage_dir(workflow)
    manifest_path = dest / 'manifest.json'
    if not manifest_path.exists():
        print(f"  No manifest in {dest} — run save_pipeline first.")
        return {'passed': False, 'reason': 'missing manifest'}

    manifest = json.loads(manifest_path.read_text())
    inputs, outputs = manifest['inputs'], manifest['outputs']

    if Test_set is None:
        job = workflow.config['job_name']
        Test_set = workflow.load_data(f"{job}_Test_set.csv")

    sample = Test_set[inputs].head(5)
    results, all_ok = [], True

    for info in manifest['pipelines']:
        label, pf = info['label'], Path(info['pipeline_file'])
        try:
            model = joblib.load(pf)
            y = np.asarray(model.predict(sample))
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            ok = y.shape[0] == len(sample) and y.shape[1] == len(outputs) and np.isfinite(y).all()
            reason = ('' if ok else
                      f'expected {(len(sample), len(outputs))}, got {y.shape}'
                      if y.shape != (len(sample), len(outputs)) else 'non-finite predictions')
        except Exception as e:
            ok, reason, y = False, f'{type(e).__name__}: {e}', None

        all_ok &= ok
        results.append({'label': label, 'passed': bool(ok), 'reason': reason})
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<20} "
              f"{'shape ' + str(y.shape) if ok else reason}")

    summary = {'passed': bool(all_ok), 'results': results, 'storage_path': str(dest)}
    workflow.metadata.update_step_data(
        {'integration_test': summary},
        ['metadata', 'Model_Storage', 'Store'],
    )
    print(f"\n  Integration test: {'PASSED' if all_ok else 'FAILED'}")
    return summary
