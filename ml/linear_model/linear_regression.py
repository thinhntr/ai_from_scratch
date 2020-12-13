# linear_regression.py


import numpy as np

from ..exceptions import NotFittedError
from ..utils import check_input, check_is_fitted


class LinearRegression:
    """
    Class to create LinearRegression model

    Attributes
    ----------
    W : np.ndarray of shape (n_features, )
        Weights of model

    b : float
        Bias

    n_features : int
        Number of dimensions of vectors that this estimator can predict
    """

    @property
    def n_features(self) -> int:
        return self.W.shape[0]

    def __init__(self):
        self.W = None
        self.b = None

    def fit(self, X, y):
        """
        Fit weights and bias based on X and y
        """
        # TODO
        raise NotImplemented

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict result using linear regression

        Parameters
        ----------
        X : np.ndarray
            input
        """
        check_is_fitted(self, ["W", "b"])
        check_input(X, self.n_features)

        return self.W @ X.T + self.b
