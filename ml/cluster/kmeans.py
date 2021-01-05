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
    @property
    def k(self) -> int:
        return len(self.centroids)

    def __init__(self, n_iters: int):
        self.centroids: List[List[float]] = []
        self.n_iters = n_iters
        self.rg__ = np.random.default_rng()

    def fit(self, X, k):
        check_input(X)
        assert (isinstance(k, int))

        self.centroids = self.rg__.choice(X, k, replace=False).tolist()

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

        padding = np.ones((n_samples, 1))
        new_X = np.hstack([X, padding]).T

        base = np.tile(np.identity(n_features), (self.k, 1))
        T = np.hstack([base, -np.array(self.centroids).ravel().reshape(-1, 1)])

        tmp = T @ new_X
        # (n * k) * m

        tmp = np.hstack(np.split(tmp, self.k))
        distances = np.linalg.norm(tmp, axis=0)
        distances = np.split(distances, self.k)
        distances = np.hstack(distances)    

        # each example lie in 1 column
        # each row i represents the distance from example in column j to cluster i
        # distances_per_example = distances.reshape(n_features, n_samples)

        # find closest cluster for each example
        y = np.argmin(distances, axis=0)

        return y

