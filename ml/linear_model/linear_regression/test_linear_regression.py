import numpy as np

from .linear_regression import LinearRegression


class TestLinearRegression:
    def test_predict(self):
        # random input
        n_samples = 10
        n_features = 15
        rg = np.random.default_rng()
        X = rg.integers(-5, 6, (n_samples, n_features))

        # expected model
        expected_W = rg.integers(-10, 21, n_features)
        expected_b = 10

        # expected output
        expected_y = expected_W @ X.T + expected_b

        a = LinearRegression()
        a.W = expected_W
        a.b = expected_b

        assert (a.predict(X) == expected_y).all()
