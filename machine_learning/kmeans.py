import numpy as np
from collections import Counter


def check_input(X):
    assert (isinstance(X, np.ndarray))
    assert (len(X.shape) == 2)


# Only use for debugging
rg = np.random.default_rng(12)


class KMeans:
    @property
    def k(self):
        """
        :return:
        Number of clusters
        """
        if self.centroids is None:
            raise RuntimeError("fit() hasn't run yet")
        return len(self.centroids)

    def __init__(self, n_iters):
        self.centroids = None
        self.n_iters = n_iters

    def fit(self, X, k):
        check_input(X)
        assert (isinstance(k, int))

        self.centroids = rg.choice(X, k, replace=False).tolist()

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
        assert (self.centroids is not None)

        n_examples = X.shape[0]
        n_features = X.shape[1]

        padding = np.ones((n_examples, 1))
        new_X = np.hstack([X, padding]).T

        base = np.tile(np.identity(n_features), (self.k, 1))
        T = np.hstack([base, -np.array(self.centroids).ravel().reshape(-1, 1)])

        tmp = T @ new_X
        tmp = np.hstack(np.split(tmp, n_features))
        distances = np.linalg.norm(tmp, axis=0)

        # each example lie in 1 column
        # each row i represents the distance from example in column j to cluster i
        distances_per_example = distances.reshape(n_features, n_examples)

        # find closest cluster for each example
        y = np.argmin(distances_per_example, axis=0)


        return y
