import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MandatorySelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, score_func, k = 30, feature_names = None, mandatory_features = None):
        self.score_func = score_func
        self.k = k
        self.feature_names = feature_names
        self.mandatory_features = mandatory_features

    def fit(self, X, y):
        X_array = np.asarray(X)

        if self.feature_names is None:
            self.feature_names_ = [f"feature_{i}" for i in range(X_array.shape[1])]
        else:
            self.feature_names_ = list(self.feature_names)

        if self.mandatory_features is None:
            self.mandatory_features_ = []
        else:
            self.mandatory_features_ = list(self.mandatory_features)

        scores, _ = self.score_func(X_array, y)
        scores = np.nan_to_num(scores, nan = -np.inf)

        mandatory_indices = [
            index for index, name in enumerate(self.feature_names_)
            if name in self.mandatory_features_
        ]

        ranked_indices = np.argsort(scores)[::-1].tolist()
        max_features = min(self.k, X_array.shape[1])

        selected_indices = []

        for index in mandatory_indices:
            if index not in selected_indices:
                selected_indices.append(index)

        for index in ranked_indices:
            if len(selected_indices) >= max_features:
                break
            if index not in selected_indices:
                selected_indices.append(index)

        self.selected_indices_ = np.array(sorted(selected_indices))
        self.selected_features_ = [
            self.feature_names_[index] for index in self.selected_indices_
        ]

        return self

    def transform(self, X):
        X_array = np.asarray(X)
        return X_array[:, self.selected_indices_]

    def get_selected_features(self):
        return self.selected_features_