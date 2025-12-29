# src/tuning.py
import pandas as pd
from typing import Tuple

from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
import xgboost as xgb

from config import SEED


def tune_xgboost(
    X,
    y,
    n_iter: int = 30,
    cv_splits: int = 3,
    n_jobs: int = -1,
    random_state: int = SEED,
) -> Tuple[xgb.XGBRegressor, pd.DataFrame]:
    """
    RandomizedSearchCV tuning for XGBoost.
    Returns best model and cv_results_ as DataFrame.
    """
    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    param_dist = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0.0, 0.1, 0.2],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state,
    )
    search.fit(X, y)

    best_model = search.best_estimator_
    cv_results = pd.DataFrame(search.cv_results_)
    return best_model, cv_results


def tune_lightgbm(
    X,
    y,
    n_iter: int = 30,
    cv_splits: int = 3,
    n_jobs: int = -1,
    random_state: int = SEED,
) -> Tuple[lgb.LGBMRegressor, pd.DataFrame]:
    """
    RandomizedSearchCV tuning for LightGBM.
    """
    base = lgb.LGBMRegressor(
        n_estimators=1000,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    param_dist = {
        "num_leaves": [31, 63, 127, 255],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_depth": [-1, 5, 7, 9],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.0, 0.1, 0.5],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state,
    )
    search.fit(X, y)

    best_model = search.best_estimator_
    cv_results = pd.DataFrame(search.cv_results_)
    return best_model, cv_results
