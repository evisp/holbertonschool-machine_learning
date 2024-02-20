#!/usr/bin/env python3
"""
 calculates the likelihood of obtaining this data 
 given various hypothetical probabilities of developing severe side effects
"""

import numpy as np 


def likelihood(x, n, P):
    """ calculates the likelihood """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for value in P:
        if value > 1 or value < 0:
            raise ValueError("All values in P must be in the range [0, 1]")

    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(x) * factorial(n-x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))

    return likelihood
