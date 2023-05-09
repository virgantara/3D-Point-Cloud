import numpy as np
import matplotlib.pyplot as plt


# define a function to compute pairwise distances between data points
def pairwise_distances(X):
    """
    Computes pairwise distances between data points.
    """
    distances = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            distances[i, j] = np.linalg.norm(X[i] - X[j])
            distances[j, i] = distances[i, j]
    return distances


# define the DBSCAN algorithm
def dbscan(X, eps, min_samples):
    """
    DBSCAN algorithm for clustering data points.
    """
    # compute pairwise distances between data points
    distances = pairwise_distances(X)

    # find all points within eps distance of each other
    neighbors = [set(np.where(distances[i] <= eps)[0]) - set([i]) for i in range(X.shape[0])]

    # initialize cluster labels and visited set
    labels = np.zeros(X.shape[0])
    visited = set()

    # perform DBSCAN algorithm
    cluster = 0
    for i in range(X.shape[0]):
        if i in visited:
            continue
        visited.add(i)
        if len(neighbors[i]) < min_samples:
            labels[i] = -1  # noise point
        else:
            labels[i] = cluster  # new cluster
            expand_cluster(i, neighbors[i], cluster, visited, labels, neighbors, min_samples, distances)
            cluster += 1
    return labels


# define a helper function to expand the cluster
def expand_cluster(point, neighbors, cluster, visited, labels, all_neighbors, min_samples, distances):
    """
    Expands the cluster to include all density-reachable points.
    """
    for neighbor in neighbors:
        if neighbor not in visited:
            visited.add(neighbor)
            if len(all_neighbors[neighbor]) >= min_samples:
                new_neighbors = all_neighbors[neighbor] & neighbors
                neighbors |= new_neighbors
        if labels[neighbor] == 0:
            labels[neighbor] = cluster
        elif labels[neighbor] == -1:
            labels[neighbor] = cluster
        else:
            continue


# generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)

# run DBSCAN algorithm
labels = dbscan(X, eps=0.5, min_samples=5)

# plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
