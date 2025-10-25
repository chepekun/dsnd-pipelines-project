from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EmptySelectorGuard(BaseEstimator, TransformerMixin):
    """Wraps downstream transformers to handle no columns passed by a selector"""

    def __init__(self, transformer):
        self.transformer = transformer
        self._is_empty_ = None
        self._wrapped_ = None

    def fit(self, X, y=None):
        self._is_empty_ = X.shape[1] == 0

        if self._is_empty_:
            return self

        return self.transformer.fit(X, y)

    def transform(self, X):
        if self._is_empty_:
            return pd.DataFrame(index=X.index)
        return self.transformer.transform(X)

    def get_feature_names_out(self, *args, **params):
        if self._is_empty_:
            return np.array([], dtype=object)
        return self.transformer.get_feature_names_out(*args, **params)


class StoreSelector(BaseEstimator, TransformerMixin):
    """Selects which columns describing the store should be used by the model"""

    def __init__(self, include_division=True, include_department=True, include_class=True):
        self.include_division = include_division
        self.include_department = include_department
        self.include_class = include_class
        self.columns_ = []
        if include_division:
            self.columns_.append("Division Name")
        if include_department:
            self.columns_.append("Department Name")
        if include_class:
            self.columns_.append("Class Name")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns_]

    def get_feature_names_out(self, *args, **params):
        return self.columns_


class BaseReviewSelector(BaseEstimator, TransformerMixin, ABC):
    """Abstract class to select processed text column from the review"""

    def __init__(self, include_char=True, include_nlp=True, include_sentiment=True):
        self.include_char = include_char
        self.include_nlp = include_nlp
        self.include_sentiment = include_sentiment
        self.columns_ = []

    @abstractmethod
    def _get_section(self) -> str: ...

    def fit(self, X, y=None):
        section = self._get_section()
        if self.include_char:
            self.columns_.extend([col for col in X.columns if col.startswith(section + "_char_")])
        if self.include_nlp:
            self.columns_.extend([col for col in X.columns if col.startswith(section + "_nlp_")])
        if self.include_sentiment:
            self.columns_.extend([col for col in X.columns if col.startswith(section + "_sentiment_")])
        return self

    def transform(self, X):
        return X[self.columns_]

    def get_feature_names_out(self, *args, **params):
        return self.columns_


class ReviewTitleSelector(BaseReviewSelector):
    """Selects processed text column from the review title"""

    def _get_section(self) -> str:
        return "title"


class ReviewTextSelector(BaseReviewSelector):
    """Selects processed text column from the review text"""

    def _get_section(self) -> str:
        return "review_text"
