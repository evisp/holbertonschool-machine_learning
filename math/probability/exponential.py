#!/usr/bin/env python3
"""Exponential class"""


e = 2.7182818285


class Exponential:
    """Class Exponential"""
    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data)/len(data))

    def pdf(self, x):
        """Calculates the PDF"""
        if x < 0:
            return 0
        return self.lambtha * e ** (-self.lambtha * x)

    def cdf(self, x):
        """Calculates the CDF"""
        if x < 0:
            return 0
        return 1 - e ** (-self.lambtha * x)
