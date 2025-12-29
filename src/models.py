# src/models.py
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb

from config import SEED


def get_models():
    """Baseline set of models for comparison."""
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),

        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            n_jobs=-1,
            random_state=SEED,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            n_jobs=-1,
            random_state=SEED,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=6,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=SEED,
        ),
    }
    return models
