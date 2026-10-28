"""Small, shared leakage-prevention utilities."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.validation import check_is_fitted


def facility_split(frame, test_size=0.2, seed=42):
    """Return positional indices; test_size refers to facilities, not rows."""
    if frame['FAC_NO'].isna().any():
        raise ValueError('FAC_NO must be present before splitting.')
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(split.split(frame, groups=frame['FAC_NO']))
    assert set(frame.iloc[train]['FAC_NO']).isdisjoint(frame.iloc[test]['FAC_NO'])
    return train, test


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Winsorize predictors at training quantiles; retain all evaluation rows."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError('Require 0 <= lower < upper <= 1.')
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, 'lower_bounds_')
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count changed after fit.')
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else
                          [f'x{i}' for i in range(self.n_features_in_)], dtype=object)
