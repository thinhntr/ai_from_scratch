from typing import Optional

from numpy import ndarray

from .exceptions import NotFittedError

__all__ = ["check_input", "check_is_fitted"]


def check_input(X, estimator_n_features: Optional[int] = None):
    """
    Check if the user's input `X` is valid

    Parameters
    ----------
    X : Any
        User's input

    estimator_n_features
        Number of features of current estimator

    Raises
    ------
    TypeError
        If user's input is not an instance of numpy.ndarray

    ValueError
        If shape of X is not (n_samples, n_features) or
        If X dimension doesn't match estimator's n_features
    """
    if not isinstance(X, ndarray):
        raise TypeError("X must be of type ndarray")

    if X.ndim != 2:
        raise ValueError("X's dimension must be 2")

    if estimator_n_features is not None and estimator_n_features != X.shape[1]:
        raise ValueError("X's dimension doesn't match estimator n_features")


def check_is_fitted(estimator, attributes=None):
    """
    Check if the current estimator has run fit() method

    Parameters
    ----------
    estimator : estimator instance

    attributes : str, List[str], Tuple[str], default = None
        attributes that the current estimator must have after running fit() method

    Raises
    ------
    NotFittedError
        if this instance hasn't run fit() method yet
    """
    if not hasattr(estimator, "fit"):
        raise TypeError("This instance is not an estimator")

    if attributes is not None:
        if not isinstance(attributes, (tuple, list)):
            attributes = [attributes]
        attrs = all([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    if not attrs:
        raise NotFittedError("This instance is not fitted")
