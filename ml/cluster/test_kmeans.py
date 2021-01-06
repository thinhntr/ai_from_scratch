from .kmeans import KMeans
import numpy as np
from sklearn.datasets import make_blobs


class TestKMeansClass:
    def test_predict1(self):
        model = KMeans()
        n_samples = 1500
        n_clusters = 5
        X, _, centers = make_blobs(n_samples=n_samples,
                                   centers=n_clusters,
                                   random_state=29,
                                   return_centers=True)
        distances = np.zeros((n_samples, n_clusters))
        for i, x in enumerate(X):
            for j, center in enumerate(centers):
                distances[i][j] = np.linalg.norm(x - center)
        y = np.argmin(distances, axis=1)

        model.centroids = centers
        my_predict = model.predict(X)

        assert my_predict.shape == y.shape
        assert my_predict.dtype == np.int
        print(my_predict)
        print(y)
        assert (my_predict == y).all()

