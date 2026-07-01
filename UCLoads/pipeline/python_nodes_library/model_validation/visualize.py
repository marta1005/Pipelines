import matplotlib.pyplot as plt
import pandas as pd
import surrogate_factory as sf
from pathlib import Path


@sf.node
def plot(workflow, Test_set, model_output):
    """Scatter plots (y_pred vs y_true) per model, plus ratio error per output."""
    outdir = Path(workflow.config['artifacts.folder'])
    models_info = workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models'])
    outputs = workflow.metadata.get_step_data(['metadata', 'Model_Selection', 'outputs'])

    for info in models_info:
        label = info['label']
        ncols = 4
        nrows = -(-len(outputs) // ncols)  # ceiling division

        # --- Scatter: y_pred vs y_true ---
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()
        fig.suptitle(f'Predicted vs True — {label}', fontsize=13)

        for i, col in enumerate(outputs):
            y_true = Test_set[col].values
            y_pred = model_output[f'{label}__{col}'].values

            ax = axes[i]
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, label='y=x')
            ax.scatter(y_true, y_pred, s=5, alpha=0.4, color='steelblue')
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
            ax.set_title(col, fontsize=10)
            ax.legend(fontsize=8)

        for j in range(len(outputs), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        scatter_file = outdir / f'scatter_{label}.png'
        plt.savefig(scatter_file, dpi=120, bbox_inches='tight')
        plt.show()
        plt.close(fig)

        # --- Ratio error: y_pred / y_true per output ---
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes2 = axes2.flatten()
        fig2.suptitle(f'Ratio y_pred / y_true — {label}', fontsize=13)

        for i, col in enumerate(outputs):
            y_true = Test_set[col].values
            y_pred = model_output[f'{label}__{col}'].values
            # avoid division by zero
            mask = y_true != 0
            ratio = y_pred[mask] / y_true[mask]

            ax = axes2[i]
            ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--', label='ratio=1')
            ax.scatter(y_true[mask], ratio, s=5, alpha=0.4, color='tomato')
            ax.set_xlabel('True')
            ax.set_ylabel('y_pred / y_true')
            ax.set_title(col, fontsize=10)
            ax.legend(fontsize=8)

        for j in range(len(outputs), len(axes2)):
            axes2[j].set_visible(False)

        plt.tight_layout()
        ratio_file = outdir / f'ratio_{label}.png'
        plt.savefig(ratio_file, dpi=120, bbox_inches='tight')
        plt.show()
        plt.close(fig2)

        print(f"[{label}] scatter → {scatter_file.name}   ratio → {ratio_file.name}")
