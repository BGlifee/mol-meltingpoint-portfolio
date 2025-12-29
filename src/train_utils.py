# src/train_utils.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    """Root-mean-squared error (RMSE)."""
    mse = mean_squared_error(y_true, y_pred)
    return float(mse ** 0.5)


def cross_validate_models(models: dict, X, y, n_splits: int = 5, random_state: int = 42):
    """
    Run K-Fold CV for multiple models.
    models: {"name": estimator, ...}
    Returns: dict -> {name: {"mean_RMSE": ..., "std_RMSE": ...}}
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    for name, model in models.items():
        scores = []
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(rmse(y_val, pred))

        results[name] = {
            "mean_RMSE": np.mean(scores),
            "std_RMSE": np.std(scores),
        }

    return results


def cross_validate_single(model, X, y, n_splits: int = 5, random_state: int = 42):
    """Convenience wrapper for a single model CV."""
    res = cross_validate_models({"model": model}, X, y, n_splits, random_state)
    return res["model"]


def append_results_table(results: dict, experiment_name: str, csv_path):
    """
    Append CV results to a CSV.
    results: dict from cross_validate_models
    experiment_name: string to tag this experiment
    csv_path: Path or str
    """
    df = pd.DataFrame(results).T
    df.insert(0, "experiment", experiment_name)
    df.index.name = "model"

    csv_path = str(csv_path)
    try:
        old = pd.read_csv(csv_path)
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv(csv_path, index=False)
    return df
