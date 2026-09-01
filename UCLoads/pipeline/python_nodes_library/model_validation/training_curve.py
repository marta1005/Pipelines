"""
Training history plot, saved as a first-class SF_9 artifact.

The curves are not in the workflow metadata — they live on the fitted
estimator — so the saved models are reloaded to read them. Producing this in
SF_9 rather than only inside the report means the figure exists for every use
case whether or not the executive summary is ever built.
"""

import json
from pathlib import Path

import surrogate_factory as sf

FILENAME = 'training_curve.png'

# Reading a curve costs a full unpickle. Iterative learners are small; the
# models that run to gigabytes are forests, which have no training history at
# all — UCLoads' RandomForest is 1.7 GB, and loading it to discover that would
# stall SF_9 and eat the memory for nothing.
MAX_LOAD_MB = 400


_READER = r'''
import json, sys
import joblib, numpy as np

model = joblib.load(sys.argv[1])

loss = getattr(model, "loss_curve_", None)
if loss is not None and len(loss):
    out = {"loss": [float(v) for v in loss],
           "val": [float(v) for v in (getattr(model, "validation_scores_", None) or [])]}
else:
    stage = [getattr(e, "train_score_", None)
             for e in (getattr(model, "estimators_", None) or [])]
    stage = [s for s in stage if s is not None and len(s)]
    if stage:
        n = min(len(s) for s in stage)
        out = {"loss": [float(np.mean([s[i] for s in stage])) for i in range(n)],
               "val": []}
    else:
        out = None

sys.stdout.write("@@CURVE@@" + json.dumps(out))
'''


def _read_curve(path):
    """
    Read one model's training history in a child process.

    Unpickling happens out-of-process on purpose. Loading the UCLoads XGBoost
    model inside a live Workflow segfaults (exit 139) — a native library
    conflict that no try/except can catch, and which would take the Jupyter
    kernel with it. A crash here costs one curve and a printed warning.

    Returns {'loss': [...], 'val': [...]}, or None.
    """
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, '-c', _READER, str(path)],
                          capture_output=True, text=True, timeout=300)
    marker = proc.stdout.find('@@CURVE@@')
    if proc.returncode != 0 or marker < 0:
        detail = (f'crashed with signal {-proc.returncode}'
                  if proc.returncode < 0 else
                  f'exit {proc.returncode}')
        tail = (proc.stderr or '').strip().splitlines()
        raise RuntimeError(f'{detail}' + (f': {tail[-1][:120]}' if tail else ''))
    return json.loads(proc.stdout[marker + len('@@CURVE@@'):])


def extract_curves(models_info):
    """
    Read the training history off each saved model.

    MLPRegressor keeps loss_curve_, and validation_scores_ when early stopping
    is on. Gradient boosting keeps a per-stage train_score_ on each inner
    estimator, which is averaged across the one-per-output members. Anything
    with no notion of training history is skipped rather than invented.
    """
    curves = {}
    for info in models_info:
        label, path = info.get('label'), info.get('file')
        if not path or not Path(path).exists():
            continue

        size_mb = Path(path).stat().st_size / 1e6
        if size_mb > MAX_LOAD_MB:
            print(f"  {label}: {size_mb:,.0f} MB, over the {MAX_LOAD_MB} MB load "
                  f"limit — skipped (a model this large is a forest, which "
                  f"records no training history)")
            continue

        try:
            curve = _read_curve(path)
        except Exception as e:
            print(f"  {label}: could not be read — {e}")
            continue

        if curve is None:
            print(f"  {label}: no training history available — skipped")
        else:
            curves[label] = curve

    return curves


def render(curves, dest: Path):
    """
    Draw the curves to `dest` with validationlib's own training_curves_plot, so
    the report matches validation_output.html instead of carrying a separate,
    differently-styled reimplementation.

    Returns the path, or None if there is nothing to draw.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from validationlib.models.validation import training_curves_plot

    if not curves:
        return None

    # One metric per model, keyed by label, which is how the function expects
    # its two dictionaries.
    training = {label: c['loss'] for label, c in curves.items()}
    validation = {label: c['val'] for label, c in curves.items() if c['val']}

    # Library defaults (figHsize=7, aspect 1.5) give a poster-sized panel per
    # model; this keeps it a modest inset in the report.
    # No figure-level title: at this size the library's suptitle lands on top
    # of the per-panel titles, and the report section heading already names it.
    fig = training_curves_plot(
        training_metrics=training,
        validation_metrics=validation,
        plot_by_epoch=False,
        ylogscale=True,
        figHsize=4.0,
        figAspectRatio=1.5,
    )
    # The compact size also crowds the x ticks into an unreadable run
    # ("0 50 100 150 200 …"); thin them and shrink the text to match.
    from matplotlib.ticker import MaxNLocator
    for ax in fig.get_axes():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.title.set_size(9)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(7)
    fig.tight_layout()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return dest


@sf.node
def plot_training_curve(workflow):
    """Save the training history alongside the other SF_9 artifacts."""
    models_info = workflow.metadata.get_step_data(
        ['metadata', 'Model_Training', 'Models']) or []

    curves = extract_curves(models_info)
    dest = Path(workflow.config['artifacts.folder']) / FILENAME
    path = render(curves, dest)

    if path is None:
        print("  No model exposes a training history — nothing saved.")
        workflow.metadata.update_step_data(
            {'training_curve': None}, ['metadata', 'Model_Validation'])
        return None

    for label, c in curves.items():
        tail = c['loss'][-1]
        extra = f", final val R² {c['val'][-1]:.4f}" if c['val'] else ""
        print(f"  {label:<20} {len(c['loss']):>4} iterations, final loss {tail:.6g}{extra}")
    print(f"\n  saved → {path.name}")

    # Explicit path: this node name is not in the framework's process
    # mapping, so current_step would be empty — and metadata.py treats an
    # empty path as 'replace the root object'.
    workflow.metadata.update_step_data(
        {'training_curve': str(path)}, ['metadata', 'Model_Validation'])
    return str(path)
