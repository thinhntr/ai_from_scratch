# from machine_learning import kmeans
import matplotlib.pyplot as plt
import cv2
import numpy as np

rg = np.random.default_rng()


img1 = rg.integers(0, 255, size=(640, 480, 3))
img2 = rg.integers(0, 255, size=(640, 480))
img3 = cv2.imread('c:/users/thinh/pictures/mountain.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.imread('c:/users/thinh/pictures/mist-forest.jpg')
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(2, 2)
print(ax.shape)
print(type(ax))
ax[0, 0].imshow(img1)
ax[0, 1].imshow(img2, cmap='gray')
ax[1, 0].imshow(img3)
ax[1, 1].imshow(img4)

plt.show()