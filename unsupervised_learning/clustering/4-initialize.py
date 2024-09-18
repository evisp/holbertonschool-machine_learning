#!/usr/bin/env python3
"""GMM function """

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    initializes variables for a Gaussian Mixture Model
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None

    n, d = X.shape

    # priors for each cluster, initialized evenly
    phi = np.ones(k) / k

    # centroid means for each cluster, initialized with K-means
    m, _ = kmeans(X, k)

    # covariance matrices for each cluster, initialized as identity matrices
    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)

    return phi, m, S
