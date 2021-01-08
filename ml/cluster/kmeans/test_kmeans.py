from .kmeans import KMeans as MyKMeans
import numpy as np
from sklearn.datasets import make_blobs


class TestKMeansClass:
    def test_predict1(self):
        n_samples = 1500
        n_clusters = 5

        # Expected result
        X, _, centers = make_blobs(n_samples=n_samples,
                                   centers=n_clusters,
                                   random_state=29,
                                   return_centers=True)
        distances = np.zeros((n_samples, n_clusters))
        for i, x in enumerate(X):
            for j, center in enumerate(centers):
                distances[i][j] = np.linalg.norm(x - center)
        y = np.argmin(distances, axis=1)
        # Actual result
        model = MyKMeans(n_clusters)
        model.centroids = centers
        my_predict = model.predict(X)

        # Test
        assert my_predict.shape == y.shape
        assert my_predict.dtype == np.int
        assert (my_predict == y).all()

