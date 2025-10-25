import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AgeTransformer(BaseEstimator, TransformerMixin):
    """Inputes and scales the age of the reviewer"""

    def __init__(self, mean=43, min=18, max=99):
        self.mean = mean
        self.min = min
        self.max = max

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(X.fillna(self.mean).apply(lambda x: (x - self.min) / (self.max - self.min)))

    def get_feature_names_out(self, *args, **params):
        return self.columns_


class FeedbackTransformer(BaseEstimator, TransformerMixin):
    """Imputes and scales the positive feedback count column"""

    def __init__(self, fill_value=0, k=3):
        self.fill_value = fill_value
        self.k = k

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(X.fillna(self.fill_value).apply(lambda x: 1 - np.exp(-x / self.k)))

    def get_feature_names_out(self, *args, **params):
        return self.columns_
