from ml.linear_model.linear_regression import LinearRegression
import numpy as np
import pytest

from ml.linear_model.linear_regression import LinearRegression

h = LinearRegression()
h.W = np.array([1, 1])
h.b = 1


class TestLinearRegression:
    def test_check_input(self):
        X = [1, 2, 3]
        with pytest.raises(TypeError):
            h.predict(X)

        X = np.array(X)
        with pytest.raises(ValueError):
            h.predict(X)

    def test_check_input_dimension(self):
        X = np.array([[1, 1, 3], [2, 2, 1]])
        with pytest.raises(ValueError):
            h.predict(X)

    def test_predict(self):
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([3, 5, 7])
        print(h.predict(X))
        assert (h.predict(X) == y).all() == True
