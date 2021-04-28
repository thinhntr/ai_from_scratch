import numpy as np
from scipy.spatial.distance import cdist

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

    n_iters : int
        Number of iterations

    metric : str
        The distance functions can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, 
        ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, 
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
        ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’
    """

    def __init__(self, k: int, n_iters: int = 100, metric: str = 'euclidean'):
        self.n_iters = n_iters
        self.k = k
        self.metric = metric
        self.rg__ = np.random.default_rng()

    def fit(self, X):
        check_input(X)

        # self.centroids size is (k, n_samples)
        self.centroids: np.ndarray = self.rg__.choice(self.rg__.permutation(X),
                                                      self.k,
                                                      replace=False)

        self.first_centroids = np.copy(self.centroids)  # Use for debug purpose
        # Store centroids of the previous loop
        prev_centroids = np.copy(self.centroids)
        loop_count = 0
        cost = np.inf

        while loop_count < self.n_iters and cost > 0.00001:
            loop_count += 1
            y = self.predict(X)

            for i in range(self.k):
                points_in_cluster = X[y == i, :]
                if points_in_cluster.any():
                    self.centroids[i] = points_in_cluster.mean(axis=0)

            costs = np.linalg.norm(self.centroids - prev_centroids, axis=1)
            cost = costs.sum()
            prev_centroids = np.copy(self.centroids)
        return self

    def predict(self, X):
        check_input(X)
        check_is_fitted(self)

        # X_nrows, X_ncols = X.shape[:2]

        # X_tile = np.tile(X, self.k)  # shape (X_nrows, X_ncols * k)
        # centroids_tile = np.tile(self.centroids.ravel(), (X_nrows, 1))  # shape (X_nrows, X_ncols * k)

        # diff = X_tile - centroids_tile
        # diff_reshape = diff.reshape(X_nrows * self.k, X_ncols)

        # distances = np.linalg.norm(diff_reshape, axis=1)
        # # Value at row i and column j is the distance between X[i] and self.fit_X[j]
        # distances = distances.reshape(X_nrows, self.k)

        distances = cdist(X, self.centroids, metric=self.metric)

        return np.argmin(distances, axis=1)
