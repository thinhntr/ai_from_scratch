import numpy as np
from collections import Counter

def check_input(X):
    assert(isinstance(X, np.ndarray))
    assert(len(X.shape) == 2)


rg = np.random.default_rng()


class KMeans:

    def __init__(self, n_iters):
        self.centroids = None
        self.n_iters = n_iters

    def fit(self, X, k):
        check_input(X)
        assert(isinstance(k, int))
        m = X.shape[0]  # Number of training examples
        self.centroids = rg.choice(X, k, replace=False)

        

        for _ in range(self.n_iters):
            clusters = np.array(self.predict(X))

            for cluster in clusters:
                cluster.sum() / cluster.shape[0]

            for label, points in examples_of_label.items():
                cluster = np.array(points)
                n = cluster.shape[0]
                self.centroids[label] = 1 / n * cluster.mean()

        return self

    def predict(self, X):
        check_input(X)
        assert(self.centroids is not None)

        y = np.zeros_like(X.shape[0], dtype=np.uint)

        for i, x in enumerate(X):
            min_cluster = 0
            min_distance = np.inf
            for cluster, centroid in enumerate(self.centroids):
                distance = np.linalg.norm(centroid - x)
                if distance < min_distance:
                    min_distance = distance
                    min_cluster = cluster
            y[i] = min_cluster
        return y
