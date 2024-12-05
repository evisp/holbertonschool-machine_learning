#!/usr/bin/env python3
"""Poisson class"""


e = 2.7182818285


class Poisson:
    """Class Poisson"""
    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """Calculates the PMF"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        fact = 1
        for i in range(1, k + 1):
            fact = fact * i
        return (e ** -self.lambtha * self.lambtha ** k) / fact

    def cdf(self, k):
        """Calculates the CDF"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
