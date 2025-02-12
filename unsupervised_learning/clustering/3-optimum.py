#!/usr/bin/env python3
"""Optimize k Module"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance:

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters to
    check for (inclusive)
    kmax is a positive integer containing the maximum number of clusters to
    check for (inclusive)
    iterations is a positive integer containing the maximum number of
    iterations for K-means
    This function should analyze at least 2 different cluster sizes

    Returns: results, d_vars, or None, None on failure
    results is a list containing the outputs of K-means for each cluster size
    d_vars is a list containing the difference in variance from the smallest
    cluster size for each cluster size
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if kmax is not None and (not isinstance(kmax, int) or kmax < 0):
        return None, None

    if not isinstance(kmin, int) or kmin <= 0 or kmax is not None \
            and kmin >= kmax:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []

    C, clss = kmeans(X, kmin, iterations)
    results.append((C, clss))
    d_vars.append(0.0)

    small_var = variance(X, C)
    kmin += 1

    if kmax is None:
        kmax = X.shape[0]
    while kmin <= kmax:
        C, clss = kmeans(X, kmin, iterations)
        d_vars.append(small_var - variance(X, C))
        results.append((C, clss))
        kmin += 1

    return results, d_vars
