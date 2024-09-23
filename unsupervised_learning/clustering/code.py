#!/usr/bin/env python3
"""Initialize function"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
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

        centroids = np.copy(C)
        centroids_extended = C[:, np.newaxis]

        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        for c in range(k):
            if X[clss==c].size == 0:
                C[c] = np.random.uniform(X_min, X_max, size=(1, d))
            else:
                C[c] = X[clss == c].mean(axis = 0)


        centroids_extended = C[:, np.newaxis]
        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        if (centroids == C).all():
            break 

    return C, clss

    