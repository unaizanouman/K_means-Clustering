import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Convert to NumPy array
X = df.values  

# Number of clusters
k = 3

# Randomly initialize centroids
np.random.seed(42)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

max_iters = 100
for _ in range(max_iters):
    # Assign each point to the nearest centroid
    clusters = np.array([np.argmin([euclidean_distance(x, centroid) for centroid in centroids]) for x in X])

    # Compute new centroids
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # If centroids don't change, break the loop
    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

df['Cluster'] = clusters

# Visualizing the clusters (Using first two features)
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Means Clustering Using Euclidean Distance')
plt.legend()
plt.show()

