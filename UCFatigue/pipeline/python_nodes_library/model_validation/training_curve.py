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


def mean_output_variance(artifacts_dir):
    """
    Mean per-output variance of the training targets, from the yt_train.csv
    that export_validation_csvs leaves in validation_<model>/. Any model's copy
    will do — the training targets are the same. Returns a float, or None when
    no CSV exists yet.
    """
    import pandas as pd

    for csv in sorted(Path(artifacts_dir).glob('validation_*/yt_train.csv')):
        try:
            var = float(pd.read_csv(csv).var(ddof=0).mean())
        except Exception as e:
            print(f'  Var(y): could not read {csv.name} ({type(e).__name__})')
            continue
        if var > 0:
            return var
    return None


def val_scores_to_loss(scores, var_y):
    """
    Approximate validation loss from sklearn's validation_scores_.

    MLPRegressor records the R² on its internal early-stopping split, not a
    loss — plotting it against loss_curve_ (squared error / 2) puts two
    different units on one axis and fakes a huge train/validation gap. From
    R² = 1 - MSE/Var(y): approximate loss = (1 - R²) · Var(y) / 2, using the
    mean output variance of the training targets (sklearn's uniform-average
    R² does the same averaging). Clipped away from zero for the log axis.
    """
    return [max((1.0 - r) * var_y / 2.0, 1e-12) for r in scores]


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


def render(curves, dest: Path, var_y=None):
    """
    Draw the curves to `dest` with validationlib's own training_curves_plot, so
    the report matches validation_output.html instead of carrying a separate,
    differently-styled reimplementation.

    validation_scores_ is an R², not a loss; it is converted to an approximate
    validation loss via val_scores_to_loss when `var_y` is known, so both
    curves share one unit. Without var_y the validation curve is omitted
    rather than drawn in the wrong unit.

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
    validation = {}
    for label, c in curves.items():
        if not c['val']:
            continue
        if var_y is None:
            print(f"  {label}: validation R² left off the plot — Var(y) unavailable, "
                  f"so it cannot be converted to a loss comparable with training")
            continue
        validation[label] = val_scores_to_loss(c['val'], var_y)
        print(f"  {label}: validation R² converted to approximate loss "
              f"via (1-R²)·Var(y)/2, mean Var(y) = {var_y:.6g}")

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
    var_y = mean_output_variance(workflow.config['artifacts.folder'])
    path = render(curves, dest, var_y=var_y)

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
