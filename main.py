#encoding: utf-8
import cv2
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def exibir_clusters(sc, pixels):
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1])
    labels = sc.labels_

    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=labels.astype(int), edgecolor="k")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("R value")
    ax.set_ylabel("G value")
    ax.set_zlabel("B value")
    ax.dist = 12

    fig.add_axes(ax)
    plt.show()

def redesenhar_clusterizado(img, sc, lista_dados):
    _, axs = plt.subplots(1, 2)
    img3 = np.zeros((img.shape[0],img.shape[1],3),np.float64)
    COLORS = sc.cluster_centers_.astype(np.uint8)

    print(Counter(sc.labels_).most_common(10))
    for i,classe in enumerate(sc.labels_):
        pixel = np.array(COLORS[classe])
        row, col = lista_dados[i]
        img3[row][col] = pixel 
        
    axs[0].imshow(img3.astype(np.uint8))
    axs[1].imshow(img)
    plt.show()


img = cv2.imread('img/4303.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
#img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
IMG_WIDTH = img.shape[0]
IMG_HEIGHT = img.shape[1]


pixels = []
pos = []
for row in range(IMG_WIDTH):
    for col in range(IMG_HEIGHT):
        vector = img[row][col]
        pos.append((row,col))
        pixels.append((vector).astype(np.uint8))

sc = KMeans(random_state=0, n_clusters=8)

pixels = np.array(pixels)
sc.fit_predict(pixels)  

#exibir_clusters(sc, pixels)
print(sc.cluster_centers_)
redesenhar_clusterizado(img, sc, pos)