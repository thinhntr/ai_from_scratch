from typing import List

import numpy as np

from ml.utils import check_input, check_is_fitted


# Only use for debugging
# rg = np.random.default_rng(12)


class KMeans:
    """
    Class to create KMeans model

    Attributes
    ----------
    k : int
        Number of cluster

    centroids: List[np.ndarray]
        centroids' location

    n_iters : int
        Number of iterations
    """

    def __init__(self, k: int, n_iters: int = 100):
        self.n_iters = n_iters
        self.k = k
        self.rg__ = np.random.default_rng()

    def fit(self, X):
        check_input(X)

        self.centroids: List[List[float]] = self.rg__.choice(X, self.k, replace=False).tolist()

        for _ in range(self.n_iters):
            y = self.predict(X)
            for label in range(len(self.centroids)):
                points_in_cluster = X[y == label, :]
                if points_in_cluster.any():
                    self.centroids[label] = points_in_cluster.mean(axis=0)

        self.centroids = np.array(self.centroids)
        return self

    def predict(self, X):
        check_input(X)
        check_is_fitted(self)

        n_samples, n_features = X.shape

        row_ones = np.ones((1, n_samples))
        new_X = np.vstack([X.T, row_ones])  # (n_features+1, n_samples)

        k_identities = np.tile(np.identity(n_features), (self.k, 1))
        raveled = -np.array(self.centroids).ravel().reshape(-1, 1)  # convert centroids to 1-D
        T = np.hstack([k_identities, raveled])  # (n_features*k, n_features+1)

        tmp = T @ new_X  # (n_features*k, n_samples)

        tmp = np.hstack(np.split(tmp, self.k))  # (n_features, n_samples*k)
        distances = np.linalg.norm(tmp, axis=0)  # (1, n_samples*k)
        distances = distances.reshape(self.k, n_samples)

        # each example lie in 1 column
        # each row i represents the distance from example in column j to cluster i
        # find closest cluster for each example
        return np.argmin(distances, axis=0)
