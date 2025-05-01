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
        return self

    def transform(self, X):
        
        cols = X.columns
        idx  = X.index
        val  = X.values
        
        # compute std per feature
        stds = val.std(axis=0)
        # sample noise
        noise = np.random.normal(
            loc=0,
            scale=self.noise_factor * stds,
            size=val.shape
        )
        X_noisy = val + noise
        # clip to 0 bc we min max after
        
        return pd.DataFrame(X_noisy, columns=cols, index=idx)
        

