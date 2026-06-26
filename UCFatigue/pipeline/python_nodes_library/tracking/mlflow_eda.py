"""
MLflow EDA logger — logs a full data/experiment report as a dedicated run.

Logged per run:
  params   — dataset sizes, split %, feature names, random seed
  metrics  — per-feature stats, split quality (VTP), KS p-values, R²/Q90/MAE
  artifacts/eda/
    input_distributions.png   — Train vs Test histogram per numeric input
    output_distributions.png  — Train vs Test histogram per output
    correlation_matrix.png    — input feature Pearson correlation heatmap
    pca_coverage.png          — Train/Val/Test in 2-D PCA space
    feature_summary.csv       — mean/std/min/max/missing for all columns
    validation_results.csv    — R², Q90, MAE, pass/fail per model × output
"""

import os
import re
import tempfile
import warnings
import matplotlib
matplotlib.use('Agg')   # non-interactive: safe in papermill / headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mk(name):
    """Sanitize a string to a valid MLflow metric/param key."""
    return re.sub(r'[^a-zA-Z0-9_.:\- /]', '_', str(name))


def log_eda(workflow, Train_set, Val_set, Test_set,
            model_output, train_output, metrics,
            split_result=None, ks_results=None):
    """Open a dedicated MLflow run and log EDA + validation data."""
    try:
        import mlflow
    except ImportError:
        print("[EDA] MLflow not installed — skipping")
        return

    tracker_tool = workflow.config.get('tracker', {}).get('tool', 'default')
    if tracker_tool != 'mlflow':
        print("[EDA] Tracker is not mlflow — skipping EDA run")
        return

    from surrogate_factory.plugins.trackers.mlflow_tracking import setup_mlflow
    setup_mlflow(workflow.config)  # idempotent (memoized)

    experiment_name = workflow.config['job_name']
    try:
        mlflow.create_experiment(experiment_name)
    except Exception:
        pass
    mlflow.set_experiment(experiment_name)

    ms      = workflow.metadata.get_step_data(['metadata', 'Model_Selection'])
    inputs  = ms['inputs']
    outputs = ms['outputs']
    models_info = workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models'])
    labels  = [m['label'] for m in models_info]

    # Numeric inputs only (categorical like FLAP are excluded from stat plots)
    num_inputs = Train_set[inputs].select_dtypes(include='number').columns.tolist()

    n_tr = len(Train_set)
    n_vl = len(Val_set)
    n_ts = len(Test_set)
    n_total = n_tr + n_vl + n_ts

    with mlflow.start_run(run_name='EDA_Data_Report'):
        tmpdir = tempfile.mkdtemp()

        # ── 1. Dataset size params ──────────────────────────────────────────
        mlflow.log_params({
            'n_train':    n_tr,
            'n_val':      n_vl,
            'n_test':     n_ts,
            'n_total':    n_total,
            'train_pct':  f"{100 * n_tr / n_total:.1f}%",
            'val_pct':    f"{100 * n_vl / n_total:.1f}%",
            'test_pct':   f"{100 * n_ts / n_total:.1f}%",
            'n_inputs':   len(inputs),
            'n_outputs':  len(outputs),
            'inputs':     str(inputs),
            'outputs':    str(outputs),
            'models':     str(labels),
        })

        # ── 2. Per-feature statistics (metrics with no step) ───────────────
        for col in num_inputs:
            vals = pd.concat([Train_set[col], Val_set[col], Test_set[col]])
            c = _mk(col)
            mlflow.log_metrics({
                f'stat_{c}_mean':    float(vals.mean()),
                f'stat_{c}_std':     float(vals.std()),
                f'stat_{c}_min':     float(vals.min()),
                f'stat_{c}_max':     float(vals.max()),
                f'stat_{c}_missing': int(vals.isna().sum()),
            })

        for col in outputs:
            vals = pd.concat([Train_set[col], Val_set[col], Test_set[col]])
            c = _mk(col)
            mlflow.log_metrics({
                f'stat_{c}_mean': float(vals.mean()),
                f'stat_{c}_std':  float(vals.std()),
                f'stat_{c}_min':  float(vals.min()),
                f'stat_{c}_max':  float(vals.max()),
            })

        # ── 3. Split quality (VTP) ─────────────────────────────────────────
        if split_result is not None:
            fields = {
                'split_residual_voxel_prop': split_result.residual_voxel_proportion,
                'split_valid_test_prop':     split_result.valid_test_proportion,
                'split_phacking_prop':       split_result.phacking_test_proportion,
                'split_isolated_test_prop':  split_result.isolated_test_proportion,
                'split_chi2_pvalue':         split_result.chi_squared_pvalue,
            }
            mlflow.log_metrics({k: float(v) for k, v in fields.items()
                                if v is not None})

        # ── 4. KS test p-values ────────────────────────────────────────────
        if ks_results is not None:
            for lbl, out_ks in ks_results.items():
                for col, pval in out_ks.items():
                    mlflow.log_metric(f'ks_{_mk(lbl)}_{_mk(col)}', float(pval))

        # ── 5. Validation metrics ──────────────────────────────────────────
        if metrics is not None:
            for lbl, out_metrics in metrics.items():
                for col, m in out_metrics.items():
                    for key in ('R2', 'quantile90', 'mean_absolute_error'):
                        val = m.get(key)
                        if val is not None:
                            mlflow.log_metric(f'{key}_{_mk(lbl)}_{_mk(col)}', float(val))

        # ── 6. INPUT distributions plot ────────────────────────────────────
        ncols = 4
        nrows = -(-len(num_inputs) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, col in enumerate(num_inputs):
            axes[i].hist(Train_set[col].dropna(), bins=25, alpha=0.6,
                         label=f'Train ({n_tr})', color='steelblue', density=True)
            axes[i].hist(Test_set[col].dropna(),  bins=25, alpha=0.6,
                         label=f'Test ({n_ts})',  color='tomato',    density=True)
            axes[i].set_title(col, fontsize=9)
            if i == 0:
                axes[i].legend(fontsize=7)
        for j in range(len(num_inputs), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle('Input Feature Distribution — Train vs Test', fontsize=11)
        plt.tight_layout()
        _save_artifact(fig, tmpdir, 'input_distributions.png', 'eda')
        _log_artifact(os.path.join(tmpdir, 'input_distributions.png'), 'eda')

        # ── 7. OUTPUT distributions plot ───────────────────────────────────
        nrows2 = -(-len(outputs) // ncols)
        fig, axes = plt.subplots(nrows2, ncols, figsize=(5 * ncols, 3 * nrows2))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, col in enumerate(outputs):
            axes[i].hist(Train_set[col].dropna(), bins=25, alpha=0.6,
                         label=f'Train', color='steelblue', density=True)
            axes[i].hist(Test_set[col].dropna(),  bins=25, alpha=0.6,
                         label=f'Test',  color='tomato',    density=True)
            axes[i].set_title(col, fontsize=9)
            if i == 0:
                axes[i].legend(fontsize=7)
        for j in range(len(outputs), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle('Output Distribution — Train vs Test', fontsize=11)
        plt.tight_layout()
        _save_artifact(fig, tmpdir, 'output_distributions.png', 'eda')
        _log_artifact(os.path.join(tmpdir, 'output_distributions.png'), 'eda')

        # ── 8. CORRELATION matrix ──────────────────────────────────────────
        all_num = pd.concat([Train_set[num_inputs], Val_set[num_inputs],
                             Test_set[num_inputs]], ignore_index=True)
        corr   = all_num.corr()
        labels_short = [c[:10] for c in num_inputs]
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(num_inputs)))
        ax.set_yticks(range(len(num_inputs)))
        ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels_short, fontsize=8)
        for r in range(len(num_inputs)):
            for c_idx in range(len(num_inputs)):
                ax.text(c_idx, r, f"{corr.iloc[r, c_idx]:.2f}",
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(corr.iloc[r, c_idx]) > 0.6 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title('Numeric Input Feature Correlations (full dataset)')
        plt.tight_layout()
        _save_artifact(fig, tmpdir, 'correlation_matrix.png', 'eda')
        _log_artifact(os.path.join(tmpdir, 'correlation_matrix.png'), 'eda')

        # ── 9. PCA coverage scatter ────────────────────────────────────────
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca.fit(all_num)
            trPC = pca.transform(Train_set[num_inputs])
            vlPC = pca.transform(Val_set[num_inputs])
            tsPC = pca.transform(Test_set[num_inputs])

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(trPC[:, 0], trPC[:, 1], s=6,  alpha=0.4,
                       label=f'Train ({n_tr})', color='steelblue')
            ax.scatter(vlPC[:, 0], vlPC[:, 1], s=10, alpha=0.7,
                       label=f'Val ({n_vl})',   color='orange', marker='s')
            ax.scatter(tsPC[:, 0], tsPC[:, 1], s=10, alpha=0.7,
                       label=f'Test ({n_ts})',  color='tomato', marker='^')
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
            ax.set_title('Train / Val / Test Coverage — PCA of numeric inputs')
            ax.legend(fontsize=8)
            plt.tight_layout()
            _save_artifact(fig, tmpdir, 'pca_coverage.png', 'eda')
            _log_artifact(os.path.join(tmpdir, 'pca_coverage.png'), 'eda')
        except Exception as e:
            warnings.warn(f"PCA plot skipped: {e}")

        # ── 10. Feature summary CSV ────────────────────────────────────────
        rows = []
        for col in num_inputs + outputs:
            tag = 'input' if col in inputs else 'output'
            tr = Train_set[col]
            rows.append({
                'column':  col,
                'type':    tag,
                'mean':    tr.mean(),
                'std':     tr.std(),
                'min':     tr.min(),
                'max':     tr.max(),
                'missing': tr.isna().sum(),
                'n_train': n_tr,
                'n_val':   n_vl,
                'n_test':  n_ts,
            })
        feat_csv = os.path.join(tmpdir, 'feature_summary.csv')
        pd.DataFrame(rows).to_csv(feat_csv, index=False, float_format='%.4f')
        _log_artifact(feat_csv, 'eda')

        # ── 11. Validation results CSV ─────────────────────────────────────
        if metrics is not None:
            val_rows = []
            vr_list = workflow.metadata.get_step_data(
                ['metadata', 'Model_Validation', 'validation_results']) or []
            for vr in vr_list:
                base = {'output': vr['output'], 'metric': vr['metric'],
                        'target': vr['target']}
                for lbl, res in vr.get('models', {}).items():
                    base[f'{lbl}_score']  = res.get('score')
                    base[f'{lbl}_passed'] = res.get('passed')
                val_rows.append(base)
            val_csv = os.path.join(tmpdir, 'validation_results.csv')
            pd.DataFrame(val_rows).to_csv(val_csv, index=False)
            _log_artifact(val_csv, 'eda')

        print(f"[EDA] Run logged: {mlflow.active_run().info.run_id}")


def _save_artifact(fig, tmpdir, filename, folder):
    path = os.path.join(tmpdir, filename)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def _log_artifact(path, folder):
    """Log artifact to MLflow, silently skipping if the backend is unavailable."""
    try:
        import mlflow as _mlflow
        _mlflow.log_artifact(path, folder)
    except Exception as e:
        warnings.warn(f"[EDA] Artifact not logged ({os.path.basename(path)}): {e}")
