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
pipe_path   = sys.argv[2] if len(sys.argv) > 2 else ""
val_set_csv = sys.argv[3] if len(sys.argv) > 3 else ""  # raw SF_4 Val_set (all columns)
yt_cols_csv = sys.argv[4] if len(sys.argv) > 4 else ""  # names the output columns

out = None
loss = getattr(model, "loss_curve_", None)
if loss is not None and len(loss):
    # MLP: real per-epoch validation loss when the estimator recorded one
    # (MLPRegressorValCurve); the per-epoch R2 stays as fallback material.
    out = {"loss":     [float(v) for v in loss],
           "val_loss": [float(v) for v in (getattr(model, "validation_loss_curve_", None) or [])],
           "val_r2":   [float(v) for v in (getattr(model, "validation_scores_", None) or [])]}
else:
    ests = list(getattr(model, "estimators_", None) or [])
    stage = [getattr(e, "train_score_", None) for e in ests]
    stage = [s for s in stage if s is not None and len(s)]
    if stage:
        n = min(len(s) for s in stage)
        out = {"loss": [float(np.mean([s[i] for s in stage])) for i in range(n)],
               "val_loss": [], "val_r2": []}
        # Gradient boosting: the REAL per-stage validation loss, computed by
        # replaying every stage on the pipeline's validation split. Same unit
        # as train_score_ (per-stage squared-error deviance).
        if pipe_path and val_set_csv and yt_cols_csv and all(hasattr(e, "staged_predict") for e in ests):
            try:
                import pandas as pd
                pipe = joblib.load(pipe_path)
                pre = pipe.named_steps["preprocessor"]
                vs = pd.read_csv(val_set_csv)
                ycols = list(pd.read_csv(yt_cols_csv, nrows=0).columns)
                Xt = pre.transform(vs[list(pre.feature_names_in_)])
                yv = vs[ycols].values
                if yv.shape[1] == len(ests):
                    mse = np.zeros(n)
                    for k, e in enumerate(ests):
                        preds = np.stack(list(e.staged_predict(Xt))[:n])
                        mse += np.mean((preds - yv[:, k][None, :]) ** 2, axis=1)
                    out["val_loss"] = [float(v) for v in mse / len(ests)]
            except Exception as e:
                print("staged val loss failed: " + type(e).__name__ + ": " + str(e)[:100],
                      file=sys.stderr)

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


def _read_curve(path, extra_args=()):
    """
    Read one model's training history in a child process.

    Unpickling happens out-of-process on purpose. Loading the UCLoads XGBoost
    model inside a live Workflow segfaults (exit 139) — a native library
    conflict that no try/except can catch, and which would take the Jupyter
    kernel with it. A crash here costs one curve and a printed warning.

    Returns {'loss': [...], 'val_loss': [...], 'val_r2': [...]}, or None.
    """
    import os
    import subprocess
    import sys

    # The child must be able to unpickle custom estimator classes
    # (model_training.estimators.MLPRegressorValCurve).
    env = dict(os.environ)
    lib_root = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = os.pathsep.join(
        [lib_root] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))

    proc = subprocess.run([sys.executable, '-c', _READER, str(path),
                           *map(str, extra_args)],
                          capture_output=True, text=True, timeout=600, env=env)
    marker = proc.stdout.find('@@CURVE@@')
    if proc.returncode != 0 or marker < 0:
        detail = (f'crashed with signal {-proc.returncode}'
                  if proc.returncode < 0 else
                  f'exit {proc.returncode}')
        tail = (proc.stderr or '').strip().splitlines()
        raise RuntimeError(f'{detail}' + (f': {tail[-1][:120]}' if tail else ''))
    return json.loads(proc.stdout[marker + len('@@CURVE@@'):])


def extract_curves(models_info, artifacts_dir=None):
    """
    Read the training history off each saved model.

    MLPRegressorValCurve records the real per-epoch validation loss; a plain
    MLPRegressor leaves only per-epoch R² (approximated later). Gradient
    boosting keeps a per-stage train_score_ on each inner estimator, and with
    `artifacts_dir` its real per-stage validation loss is replayed on the
    validation split (validation_*/x_val.csv + yt_val.csv). Anything with no
    notion of training history is skipped rather than invented.
    """
    # Validation split for the gradient-boosting staged validation loss: the
    # raw SF_4 Val_set (validation_*/x_val.csv can lack raw columns such as
    # categorical inputs), plus any yt_val.csv to name the output columns.
    val_set = yt_cols = None
    if artifacts_dir:
        art = Path(artifacts_dir)
        raw = sorted(art.parent.glob('*_Val_set.csv'))
        named = sorted(art.glob('validation_*/yt_val.csv'))
        if raw and named:
            val_set, yt_cols = raw[0], named[0]

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

        p = Path(path)
        pipe = p.with_name(p.name.replace('model_', 'pipeline_', 1)).with_suffix('.pkl')
        extra = (pipe, val_set, yt_cols) if (pipe.exists() and val_set is not None) else ()

        try:
            curve = _read_curve(path, extra)
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
        if c.get('val_loss'):
            validation[label] = c['val_loss']
            print(f"  {label}: real validation loss recorded per iteration")
        elif c.get('val_r2'):
            if var_y is None:
                print(f"  {label}: validation R² left off the plot — Var(y) unavailable, "
                      f"so it cannot be converted to a loss comparable with training")
                continue
            validation[label] = val_scores_to_loss(c['val_r2'], var_y)
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

    curves = extract_curves(models_info,
                            artifacts_dir=workflow.config['artifacts.folder'])
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
        if c.get('val_loss'):
            extra = f", final val loss {c['val_loss'][-1]:.6g}"
        elif c.get('val_r2'):
            extra = f", final val R² {c['val_r2'][-1]:.4f}"
        else:
            extra = ""
        print(f"  {label:<20} {len(c['loss']):>4} iterations, final loss {tail:.6g}{extra}")
    print(f"\n  saved → {path.name}")

    # Explicit path: this node name is not in the framework's process
    # mapping, so current_step would be empty — and metadata.py treats an
    # empty path as 'replace the root object'.
    workflow.metadata.update_step_data(
        {'training_curve': str(path)}, ['metadata', 'Model_Validation'])
    return str(path)
