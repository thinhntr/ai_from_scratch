import numpy as np


class KNN:
    """
    Class to create k-nearest neighbors model
    """

    @property
    def n_sample_fit_(self) -> int:
        """Number of samples in the fitted data"""
        return self.fit_X.shape[0]

    @property
    def classes_(self) -> np.ndarray:
        """Class labels known to the classifier"""
        return np.unique(self.fit_y)
    
    @property
    def nclass_(self) -> int:
        """Number of classes"""
        return len(self.classes_)

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.rg__ = np.random.default_rng()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Save X and y to this instance
        """
        self.fit_X = X
        self.fit_y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict results using fitted knn
        """
        if not hasattr(self, "fit_X"):
            raise RuntimeError("This instance has not called 'fit' yet")

        X_nrows, X_ncols = X.shape[:2]

        X_tile = np.tile(X, self.n_sample_fit_)
        fit_X_tile = np.tile(self.fit_X.ravel(), (X_nrows, 1))

        diff = X_tile - fit_X_tile
        diff_reshape = diff.reshape(X_nrows * self.n_sample_fit_, X_ncols)

        distances = np.linalg.norm(diff_reshape, axis=1)
        # Value at row i and column j is the distance between X[i] and self.fit_X[j]
        distances = distances.reshape(X_nrows, self.n_sample_fit_)

        k = self.n_neighbors
        kbest_indices = np.argpartition(distances, k)[:, :k]
        kbest_labels = np.take_along_axis(
            np.tile(self.fit_y, (X_nrows, 1)), kbest_indices, axis=1)
        
        kbest_sum = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.nclass_), 1, kbest_labels)
        kbest_label = np.argmax(kbest_sum, axis=1)
        
        return kbest_label
