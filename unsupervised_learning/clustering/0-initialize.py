#!/usr/bin/env python3
"""Initialize K-means Module"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means:

    X is a numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate uniform
    distribution along each dimension in d:
    The minimum values for the distribution should be the minimum values of X
    along each dimension in d
    The maximum values for the distribution should be the maximum values of X
    along each dimension in d

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    _, d = X.shape

    if not isinstance(k, int) or k <= 0:
        return None

    centroids = np.random.uniform(low=np.min(
        X, axis=0), high=np.max(X, axis=0), size=(k, d))

    return centroids
