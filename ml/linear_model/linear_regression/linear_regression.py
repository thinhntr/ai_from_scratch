# linear_regression.py


import numpy as np

from ml.utils import check_input, check_is_fitted


class LinearRegression:
    """
    Class to create LinearRegression model

    Attributes
    ----------
    alpha: float
        learning rate

    W : np.ndarray of shape (n_features, )
        Weights of model

    b : float
        Bias
    """

    @property
    def n_features(self) -> int:
        """Number of dimensions of vectors that this estimator can predict"""
        return self.W.shape[0]

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.W = None
        self.b = None
        self.rg__ = np.random.default_rng()

    def cost__(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate cost of current weights and bias with respect to X and y

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            input
        y : np.ndarray of shape (n_samples, )
            expected output

        Returns
        -------
        float
            Cost of this model with respect to X and y.
            Formula
            .. math:: \frac{1}{2m} \sum_{i=1}^{m} (\hat{y} - y)^2
        """
        n_samples = X.shape[0]
        y_hat = self.predict(X)  # Our predict result
        return (1 / (2 * n_samples)) * ((y_hat - y) ** 2).sum()

    def derivative_cost__(self, X, y):
        n_samples = X.shape[0]
        y_hat = self.predict(X)
        diff = y_hat - y
        derivative = (1 / n_samples) * (diff @ X)
        return derivative, diff.mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit weights and bias based on X and y
        """
        check_input(X)
        n_samples, n_features = X.shape

        self.W = self.rg__.random(n_features)
        self.b = self.rg__.random()

        loop_count = 0
        prev_J = 0
        J = np.inf
        j_vals = []

        while np.abs(J - prev_J) > 0.00001 and loop_count < 200:
            loop_count += 1
            prev_J = J
            derivative_w, derivative_b = self.derivative_cost__(X, y)
            self.W = self.W - self.alpha * derivative_w
            self.b = self.b - self.alpha * derivative_b
            J = self.cost__(X, y)
            j_vals.append(J)

        return j_vals, loop_count

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict result using linear regression

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            input

        Returns
        -------
        np.ndarray
            Result array of shape (n_samples, )
        """
        check_is_fitted(self, ["W", "b"])
        check_input(X, self.n_features)

        return self.W @ X.T + self.b
