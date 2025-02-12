#!/usr/bin/env python3
"""Performs K-means Module"""
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

    _, d = X.shape

    centroids = np.random.uniform(low=np.min(
        X, axis=0), high=np.max(X, axis=0), size=(k, d))

    return centroids


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations that should be performed
    If no change in the cluster centroids occurs between iterations, your
    function should return
    Initialize the cluster centroids using a multivariate uniform distribution
    (based on0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
    its centroid
    Returns: C, clss, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid means for each
    cluster
    clss is a numpy.ndarray of shape (n,) containing the index of the cluster
    in C that each data point belongs to
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = initialize(X, k)

    for _ in range(iterations):
        C_prev = np.copy(C)
        sum_cluster_points = np.zeros_like(C)
        n_cluster_points = np.zeros((k, 1))

        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=-1)
        clss = np.argmin(distances, axis=1)

        for i in range(k):
            cluster_points = X[clss == i]
            if len(cluster_points) == 0:
                C[i] = initialize(X, 1)[0]
            else:
                sum_cluster_points[i] = np.sum(cluster_points, axis=0)
                n_cluster_points[i] = cluster_points.shape[0]

        non_empty_clusters = n_cluster_points.flatten() != 0
        C[non_empty_clusters] = sum_cluster_points[non_empty_clusters] / \
            n_cluster_points[non_empty_clusters]

        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=-1)
        clss = np.argmin(distances, axis=1)

        if np.array_equal(C, C_prev):
            return C, clss

    return C, clss
