import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from typing import List


class GaussianNoiseInjector(BaseEstimator, TransformerMixin):
    
    ### Add Gaussian noise into features proportionally to each feature's std.
    
    def __init__(self, noise_factor=0.01):
        self.noise_factor = noise_factor

    def fit(self, X, y=None):
        # nothing to learn, but store RNG
        self._rng = np.random.RandomState()
        return self

    def transform(self, X):
        # work on numpy arrays (or convert DataFrame→array, then back)
        is_df = hasattr(X, "values")
        if is_df:
            cols = X.columns
            idx  = X.index
            arr  = X.values
        else:
            arr = np.asarray(X)
        
        # compute std per feature
        stds = arr.std(axis=0)
        # sample noise
        noise = self._rng.normal(
            loc=0,
            scale=self.noise_factor * stds,
            size=arr.shape
        )
        X_noisy = arr + noise
        # clip to 0 bc we min max after
        if is_df:
            return type(X)(X_noisy, columns=cols, index=idx)
        else:
            return X_noisy

class HighCorrelationDropper(BaseEstimator, TransformerMixin):
    """Remove one of each pair of highly correlated features (|p| ≥ ``threshold``)."""

    def __init__(self, threshold: float = 0.90, method: str = "pearson"):
        self.threshold = threshold
        self.method = method

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 scikit‑signature
        corr = X.corr(method=self.method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.to_drop_: List[str] = [c for c in upper.columns if any(upper[c] >= self.threshold)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "to_drop_")
        return X.drop(columns=self.to_drop_, errors="ignore")

    # Handy accessor
    def get_support(self) -> List[str]:
        check_is_fitted(self, "to_drop_")
        return self.to_drop_