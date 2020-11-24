
import matplotlib.pyplot as plt
import numpy as np

from machine_learning.kmeans import KMeans
from sklearn import cluster

if __name__ == '__main__':
    # Setup test
    rg = np.random.default_rng(29)
    points = rg.random(size=(150, 3))

    # Model config
    n_iters = 50
    k = 4

    # My model
    model = KMeans(n_iters=n_iters)
    model.fit(points, k)
    my_centers = model.centroids
    print(my_centers)

    # sklearn model
    sk_model = cluster.KMeans(k)
    sk_model.fit(points)
    sk_centers = sk_model.cluster_centers_
    print(sk_centers)

    print(((sk_centers - my_centers) ** 2).sum() / (2 * sk_centers.size))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(my_centers[:, 0], my_centers[:, 1], my_centers[:, 2], color='green')
    ax.scatter(sk_centers[:, 0], sk_centers[:, 1], sk_centers[:, 2], color='red')
    plt.show()
