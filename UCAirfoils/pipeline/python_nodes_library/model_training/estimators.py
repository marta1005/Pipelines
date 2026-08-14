import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


class _LogTargetRegressor:
    """
    Fit selected target columns in log space, predict back in linear space.

    CD is strictly positive and spans roughly a decade. Fitted on the raw value
    a squared-error model spends its capacity where CD is large, leaving a big
    *relative* error on the small values — which is exactly what the Q90
    requirement measures. Fitting log(CD) makes the loss proportional to
    relative error instead, and exp() on the way out keeps every downstream
    stage (validation, deployment, storage) working in physical units.

    Targets are named, not positional, so reordering the outputs cannot
    silently transform the wrong column.
    """

    def __init__(self, estimator, log_targets):
        self.estimator = estimator
        self.log_targets = list(log_targets)

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
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        y = np.asarray(self.estimator.predict(X), dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = y.copy()
        for i in self.log_idx_:
            y[:, i] = np.exp(y[:, i])
        return y

    def __getattr__(self, name):
        # Expose the inner model's fitted attributes (loss_curve_, n_iter_,
        # validation_scores_) so the MLflow logging in learn.py keeps working.
        # Guarded against recursion while unpickling, when 'estimator' is not
        # yet in __dict__.
        if name.startswith('__'):
            raise AttributeError(name)
        estimator = self.__dict__.get('estimator')
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)


def LogTargetMLPRegressor(log_targets=('CD',), **kwargs):
    """MLPRegressor that trains the named targets in log space."""
    return _LogTargetRegressor(MLPRegressor(**kwargs), log_targets)
