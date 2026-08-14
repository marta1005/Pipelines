import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class _LogTargetRegressor(BaseEstimator, RegressorMixin):
    """
    Fit selected target columns in log space, optionally standardise all
    targets, and undo both on predict.

    Two separate problems, and they interact:

    *log_targets* — CD is strictly positive and spans roughly a decade. Fitted
    on the raw value a squared-error model spends its capacity where CD is
    large, leaving a big *relative* error on the small values, which is exactly
    what the Q90 requirement measures. Fitting log(CD) makes the loss track
    relative error instead.

    *scale_targets* — a multi-output MLP minimises one shared squared error, so
    with unscaled targets the widest-ranging column dominates the gradient.
    Measured here: CL has std 0.7 against CD's 0.013, which is why CD was the
    worst output to begin with; and once CD is in log space it spans ~3.7 and
    takes over in turn, degrading the other four. Standardising every target
    puts them on equal footing.

    Targets are named, not positional, so reordering the outputs cannot
    silently transform the wrong column.
    """

    def __init__(self, estimator, log_targets, scale_targets=True):
        self.estimator = estimator
        self.log_targets = list(log_targets)
        self.scale_targets = scale_targets

    def fit(self, X, y):
        y = pd.DataFrame(y).copy()
        self.columns_ = list(y.columns)

        missing = [c for c in self.log_targets if c not in self.columns_]
        if missing:
            raise ValueError(
                f"log_targets {missing} are not among the outputs {self.columns_}"
            )

        for c in self.log_targets:
            if (y[c] <= 0).any():
                n = int((y[c] <= 0).sum())
                raise ValueError(
                    f"'{c}' has {n} non-positive value(s); it cannot be log-transformed."
                )
            y[c] = np.log(y[c])

        self.log_idx_ = [self.columns_.index(c) for c in self.log_targets]

        values = y.to_numpy(dtype=float)
        if self.scale_targets:
            self.y_scaler_ = StandardScaler().fit(values)
            values = self.y_scaler_.transform(values)
        else:
            self.y_scaler_ = None

        self.estimator.fit(X, values)
        return self

    def predict(self, X):
        y = np.asarray(self.estimator.predict(X), dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if self.y_scaler_ is not None:
            y = self.y_scaler_.inverse_transform(y)
        y = y.copy()
        for i in self.log_idx_:
            y[:, i] = np.exp(y[:, i])
        return y

    def __sklearn_is_fitted__(self):
        return hasattr(self, 'columns_')

    def __getattr__(self, name):
        # Expose the inner model's fitted attributes (loss_curve_, n_iter_,
        # validation_scores_) so the MLflow logging in learn.py keeps working.
        #
        # Dunders are deliberately not forwarded: sklearn probes for optional
        # protocol methods such as __sklearn_tags__, and answering those from
        # the wrapped MLPRegressor would misdescribe this estimator. Inheriting
        # BaseEstimator/RegressorMixin supplies them properly — without that,
        # Pipeline.predict raised AttributeError: __sklearn_tags__, which broke
        # SF_10's integration test while SF_9 stayed green because it loads the
        # bare .modl rather than the packaged Pipeline.
        #
        # The __dict__ lookup also guards against recursion while unpickling,
        # before 'estimator' has been restored.
        if name.startswith('__'):
            raise AttributeError(name)
        estimator = self.__dict__.get('estimator')
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)


def LogTargetMLPRegressor(log_targets=('CD',), scale_targets=True, **kwargs):
    """
    MLPRegressor that trains the named targets in log space and, by default,
    standardises all targets so no single output dominates the shared loss.
    Remaining keyword arguments go to MLPRegressor.
    """
    return _LogTargetRegressor(MLPRegressor(**kwargs), log_targets, scale_targets)
