# src/ensembles.py
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge


def build_stacking_regressor(base_models: dict, meta_model=None):
    """
    base_models: {"name": estimator, ...}
    meta_model: final estimator (default Ridge)
    """
    if meta_model is None:
        meta_model = Ridge(alpha=1.0)

    estimators = [(name, model) for name, model in base_models.items()]
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        n_jobs=-1,
        passthrough=False,
    )
    return stack
