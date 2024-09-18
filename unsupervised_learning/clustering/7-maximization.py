#!/usr/bin/env python3
"""Maximization Module"""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
    for each data point in each cluster
    You may use at most 1 loop

    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the updated priors for each
    cluster
    m is a numpy.ndarray of shape (k, d) containing the updated centroid means
    for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the updated covariance
    matrices for each cluster
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    n, d = X.shape

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    k, n1 = g.shape

    if n != n1 or np.abs(np.sum(g, axis=0) - 1).max() > 1e-10:
        return None, None, None

    S = np.zeros((k, d, d))

    sum_g = np.sum(g, axis=1)

    pi = sum_g / n
    m = np.dot(g, X) / sum_g[:, np.newaxis]

    for i in range(k):
        diff = X - m[i]
        weighted_diff = (g[i, :, np.newaxis] * diff).T
        S[i] = np.dot(weighted_diff, diff) / sum_g[i]

    return pi, m, S
