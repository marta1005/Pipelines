from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def MultiOutputGradientBoosting(**kwargs):
    """GradientBoostingRegressor wrapped for multioutput regression.

    sklearn's GradientBoostingRegressor is single-output only.
    MultiOutputRegressor trains one independent GBR per output column.
    This loses inter-output correlations but gives strong per-output accuracy.
    """
    return MultiOutputRegressor(GradientBoostingRegressor(**kwargs))
