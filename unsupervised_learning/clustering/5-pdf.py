#!/usr/bin/env python3
"""PDF Module"""
import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution:

    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
    should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
    distribution
    You are not allowed to use any loops
    You are not allowed to use the function numpy.diag or the method
    numpy.ndarray.diagonal

    Returns: P, or None on failure
    P is a numpy.ndarray of shape (n,) containing the PDF values for each
    data point
    All values in P should have a minimum value of 1e-300
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    d = X.shape[1]

    if not isinstance(m, np.ndarray) or len(m.shape) != 1 or \
            m.shape[0] != d:
        return None

    if not isinstance(S, np.ndarray) or len(S.shape) != 2 or \
            S.shape[0] != d or S.shape[1] != d:
        return None

    det_sqrt = np.linalg.det(S)**0.5
    inv_S = np.linalg.inv(S)

    diff = X - m
    exponent = -0.5 * np.einsum('ij,ji->i', diff @ inv_S, diff.T)

    P = (1 / ((2 * np.math.pi)**(d/2) * det_sqrt)) * np.exp(exponent)

    P = np.maximum(P, 1e-300)

    return P
