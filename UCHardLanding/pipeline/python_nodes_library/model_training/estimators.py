from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def MultiOutputGradientBoosting(**kwargs):
    """GradientBoostingRegressor wrapped for multioutput regression."""
    return MultiOutputRegressor(GradientBoostingRegressor(**kwargs))
