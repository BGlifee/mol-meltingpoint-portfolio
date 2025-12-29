import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from scipy.sparse import csr_matrix


class TabularFeatureReducer:
    """
    Pipeline for:
      1) Z-score scaling + clipping
      2) Removing highly correlated features
      3) Selecting top-N features by LightGBM importance.

    Works with both train (fit_transform) and test (transform).
    """

    def __init__(
        self,
        clip_z: float = 5.0,
        corr_threshold: float = 0.95,
        top_n: int | None = 300,
        random_state: int = 42,
    ):
        self.clip_z = clip_z
        self.corr_threshold = corr_threshold
        self.top_n = top_n
        self.random_state = random_state

        # fitted objects
        self.scaler: StandardScaler | None = None
        self.corr_keep_idx: np.ndarray | None = None
        self.lgbm_keep_idx: np.ndarray | None = None
        self.is_df: bool | None = None
        self.out_columns: list[str] | None = None

    # ---------- helpers ----------
    def _to_array(self, X):
        self.is_df = isinstance(X, pd.DataFrame)
        if self.is_df:
            self.in_columns = list(X.columns)
            return X.values
        return np.asarray(X)

    def _to_output(self, X_arr):
        if self.is_df and self.out_columns is not None:
            return pd.DataFrame(X_arr, columns=self.out_columns)
        return X_arr

    # ---------- public API ----------
    def fit_transform(self, X, y):
        """
        Fit the reducer on training data and return transformed X.
        """
        X_arr = self._to_array(X)

        # 1) scale + clip
        self.scaler = StandardScaler()
        Z = self.scaler.fit_transform(X_arr)
        Z = np.clip(Z, -self.clip_z, self.clip_z)

        # 2) drop highly correlated features
        corr = np.corrcoef(Z, rowvar=False)
        corr = np.abs(corr)
        upper = np.triu(corr, k=1)
        drop_mask = (upper > self.corr_threshold).any(axis=0)
        self.corr_keep_idx = np.where(~drop_mask)[0]
        Z_corr = Z[:, self.corr_keep_idx]

        # 3) LightGBM importance top-N
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=self.random_state,
        )
        model.fit(Z_corr, y)

        importances = model.feature_importances_
        if self.top_n is None or self.top_n >= Z_corr.shape[1]:
            self.lgbm_keep_idx = np.arange(Z_corr.shape[1])
        else:
            self.lgbm_keep_idx = np.argsort(importances)[::-1][: self.top_n]

        Z_final = Z_corr[:, self.lgbm_keep_idx]

        # store final column names (if input was DataFrame)
        if self.is_df:
            cols_after_corr = [self.in_columns[i] for i in self.corr_keep_idx]
            self.out_columns = [cols_after_corr[i] for i in self.lgbm_keep_idx]

        return self._to_output(Z_final)

    def transform(self, X):
        """
        Apply the fitted reducer to new data (test / validation).
        """
        if self.scaler is None:
            raise RuntimeError("Reducer is not fitted. Call fit_transform() first.")

        X_arr = self._to_array(X)

        Z = self.scaler.transform(X_arr)
        Z = np.clip(Z, -self.clip_z, self.clip_z)

        Z_corr = Z[:, self.corr_keep_idx]
        Z_final = Z_corr[:, self.lgbm_keep_idx]

        return self._to_output(Z_final)


class FingerprintReducer:
    """
    For Morgan fingerprint blocks (sparse):
      1) VarianceThreshold to drop almost-never-on bits
      2) Optional TruncatedSVD for dimensionality reduction.

    Use fit_transform on train, transform on test.
    """

    def __init__(
        self,
        var_threshold: float = 0.0,
        svd_components: int | None = None,
        random_state: int = 42,
    ):
        self.var_threshold = var_threshold
        self.svd_components = svd_components
        self.random_state = random_state

        self.vt: VarianceThreshold | None = None
        self.svd: TruncatedSVD | None = None

    def fit_transform(self, X_fp: csr_matrix):
        self.vt = VarianceThreshold(threshold=self.var_threshold)
        X_var = self.vt.fit_transform(X_fp)

        if self.svd_components is None:
            return X_var

        self.svd = TruncatedSVD(
            n_components=self.svd_components,
            random_state=self.random_state,
        )
        X_red = self.svd.fit_transform(X_var)
        return X_red

    def transform(self, X_fp: csr_matrix):
        if self.vt is None:
            raise RuntimeError("FingerprintReducer is not fitted. Call fit_transform() first.")

        X_var = self.vt.transform(X_fp)

        if self.svd is None:
            return X_var

        return self.svd.transform(X_var)
