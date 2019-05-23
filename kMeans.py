import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("iris.csv")
data = np.array(df)
X = data[:,:2]
y = data[:,-1]
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')  
print(kmeans.predict([[7.5,3]]))
#plt.scatter(centroids[0],centroids[1],c='k')
plt.show()

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X,y)
print(knn.predict([[7.5,3]]))