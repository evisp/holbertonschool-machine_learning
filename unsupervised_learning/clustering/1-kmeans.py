#!/usr/bin/env python3
"""K means"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    K-means on a data set
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Setting min and max values per col
    n, d = X.shape
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # Centroid
    C = np.random.uniform(X_min, X_max, size=(k, d))

    # Loop for the maximum number of iterations
    for i in range(iterations):

        # initializes k centroids by selecting them from the data points
        centroids = np.copy(C)
        centroids_extended = C[:, np.newaxis]

        # euclidean distance (dimensions of the squared distances)
        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        # an array containing the index to the nearest centroid for each point
        clss = np.argmin(distances, axis=0)

        # Assign all points to the nearest centroid
        for c in range(k):
            if X[clss == c].size == 0:  # cluster is empty
                C[c] = np.random.uniform(X_min, X_max, size=(1, d))
            else:
                C[c] = X[clss == c].mean(axis=0)

        # repeat again
        centroids_extended = C[:, np.newaxis]
        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        # if there are ano changes
        if (centroids == C).all():
            break

    return C, clss
