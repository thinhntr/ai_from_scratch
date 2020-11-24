
import matplotlib.pyplot as plt
import numpy as np

from machine_learning.kmeans import KMeans


rg = np.random.default_rng(12)

points = rg.integers(-10, 10, size=(5, 2))

model = KMeans(n_iters=20).fit(points, 2)
centers = model.centroids
print(centers)

plt.scatter(points[:, 0], points[:, 1])
plt.scatter(centers[:, 0], centers[:, 1])
plt.show()
