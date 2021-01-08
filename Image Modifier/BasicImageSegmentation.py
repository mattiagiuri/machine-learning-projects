from sklearn.cluster import KMeans
from matplotlib.image import imread
import matplotlib.pyplot as plt

initial = imread('peachone.jpg')/255

km = KMeans(n_clusters=6)
edited = initial.reshape(-1, 3)
km.fit(edited)
segmented = km.cluster_centers_[km.labels_]
segmented = segmented.reshape(initial.shape)

plt.imsave(arr=segmented, fname='peach.JPG')
