"""
MLPRegressor that also records the real validation loss per epoch.

Shared by every use case (identical copies); the SF_6/SF_7 catalog maps the
'MLPRegressor' algorithm name to this class via pipeline_config.yaml.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor


class MLPRegressorValCurve(MLPRegressor):
    """MLPRegressor that also records the real validation loss per epoch.

    sklearn keeps loss_curve_ (training loss) but, for the early-stopping
    split, only an R² per epoch (validation_scores_) — nothing in the same
    unit as the training curve. This subclass hooks the exact point where
    sklearn evaluates that split every epoch (_update_no_improvement_count)
    and records mean squared error / 2 as validation_loss_curve_, matching
    loss_curve_ up to the L2 penalty term that only the training loss carries.

    Training dynamics are untouched: same internal split, same updates, same
    stopping criterion — with the same random_state this fits the exact same
    model a plain MLPRegressor would. The *args soak up signature differences
    across sklearn versions (sample_weight arrived in the hook later); any
    failure inside the hook loses the curve, never the training run.
    """

    def fit(self, X, y):
        self.validation_loss_curve_ = []
        return super().fit(X, y)

    def _update_no_improvement_count(self, early_stopping, X_val, y_val, *args):
        if early_stopping:
            try:
                y_pred = self._predict(X_val)
                self.validation_loss_curve_.append(
                    float(np.mean((np.asarray(y_val) - y_pred) ** 2) / 2.0))
            except Exception:
                pass
        return super()._update_no_improvement_count(
            early_stopping, X_val, y_val, *args)
