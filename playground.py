import numpy as np

k = 2
centroids = [[1, 1], [4, 4]]


def predict(X: np.ndarray) -> np.ndarray:

    n_samples, n_features = X.shape

    padding = np.ones((n_samples, 1))
    new_X = np.hstack([X, padding]).T

    base = np.tile(np.identity(n_features), (k, 1))
    T = np.hstack([base, -np.array(centroids).ravel().reshape(-1, 1)])

    tmp = T @ new_X
    tmp = np.hstack(np.split(tmp, n_features))
    distances = np.linalg.norm(tmp, axis=0)

    # each example lie in 1 column
    # each row i represents the distance from example in column j to cluster i
    distances_per_example = distances.reshape(n_features, n_samples)

    # find closest cluster for each example
    y = np.argmin(distances_per_example, axis=0)

    return y


X = np.array([[0.9, 0.8], [3.8, 4.1]])
print(predict(X))