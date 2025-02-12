#!/usr/bin/env python3
"""Bayesian Information Criterion Module"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion:

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters to
    check for (inclusive)
    kmax is a positive integer containing the maximum number of clusters to
    check for (inclusive)
    If kmax is None, kmax should be set to the maximum number of clusters
    possible
    iterations is a positive integer containing the maximum number of
    iterations for the EM algorithm
    tol is a non-negative float containing the tolerance for the EM algorithm
    verbose is a boolean that determines if the EM algorithm should print
    information to the standard output

    Returns: best_k, best_result, l, b, or None, None, None, None on failure
    best_k is the best value for k based on its BIC
    best_result is tuple containing pi, m, S
    pi is a numpy.ndarray of shape (k,) containing the cluster priors for the
    best number of clusters
    m is a numpy.ndarray of shape (k, d) containing the centroid means for the
    best number of clusters
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
    for the best number of clusters
    l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log
    lh for each cluster size tested
    b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value
    for each cluster size tested
    Use: BIC = p * ln(n) - 2 * l
    p is the number of parameters required for the model
    n is the number of data points used to create the model
    l is the log lh of the model
    """

    try:
        if kmax == 1:
            return None, None, None, None
        n, d = X.shape
        if kmax is None:
            kmax = n
            if kmax >= kmin:
                return None, None, None, None

        k_history = list(range(kmin, kmax+1))
        results_history = []
        lh_history = []
        bic_history = []

        for k in range(kmin, kmax+1):
            pi, m, S, g, lh = expectation_maximization(X, k, iterations, tol,
                                                       verbose)

            if pi is None or m is None or S is None or g is None or lh is None:
                return None, None, None, None

            num_parameters = k + k * d + k * d * (d + 1) // 2 - 1
            bic = num_parameters * np.log(n) - 2 * lh

            lh_history.append(lh)
            results_history.append((pi, m, S))
            bic_history.append(bic)

        min_bic_index = np.argmin(bic_history)
        best_k = k_history[min_bic_index]
        best_result = results_history[min_bic_index]

        return best_k, best_result, np.array(lh_history), np.array(bic_history)
    except Exception:
        return None, None, None, None
