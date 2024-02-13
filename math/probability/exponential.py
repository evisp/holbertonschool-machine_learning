#!/usr/bin/env python3
"""a class Exponential that represents an exponential distribution"""


class Poisson:
    """
    a class Exponential that represents an exponential distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        class constructor
        """
        if data is None:
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            lambtha = float(len(data) / sum(data))
            self.lambtha = lambtha
