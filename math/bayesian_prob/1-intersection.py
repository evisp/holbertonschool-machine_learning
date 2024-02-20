#!/usr/bin/env python3
"""
 hat calculates the intersection of obtaining this data 
 with the various hypothetical probabilities
"""

import numpy as np 


def intersection(x, n, P, Pr):
    """ calculates the likelihood """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for idx in range(P.shape[0]):
        if P[idx] > 1 or P[idx] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[idx] > 1 or Pr[idx] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")

    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(x) * factorial(n-x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))

    intersection = likelihood * Pr
    return intersection
