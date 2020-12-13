import pytest

from machine_learning.linear_regression import LinearRegression


class TestLinearRegression:
    def test_W_before_fit(self):
        assert LinearRegression().W is None

    def test_b_before_fit(self):
        assert LinearRegression().b is None
