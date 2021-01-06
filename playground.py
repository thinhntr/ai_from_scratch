import numpy as np
from sklearn.datasets import make_blobs

from ml.cluster import KMeans

#
# def test_predict1():
#     model = KMeans()
#     n_samples = 9
#     n_clusters = 3
#     X, _, centers = make_blobs(n_samples=n_samples,
#                                centers=n_clusters,
#                                random_state=29,
#                                return_centers=True)
#     distances = np.zeros((n_samples, n_clusters))
#     for i, x in enumerate(X):
#         for j, center in enumerate(centers):
#             distances[i][j] = np.linalg.norm(x - center)
#     y = np.argmin(distances, axis=1)
#
#     model.centroids = centers
#     my_predict = model.predict(X)
#     print(my_predict == y)


if __name__ == "__main__":
    test_predict1()
