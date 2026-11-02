#!/usr/bin/env python3
"""Exponential distribution.

The exponential distribution models the waiting time until the next event
in a Poisson process (time between calls, between failures, between
arrivals). It is the continuous companion of the Poisson distribution.

It is described by a single parameter, lambtha, the expected number of
events per unit of time. Its probability density function is:

    f(x) = lambtha * e ** (-lambtha * x)    for x >= 0
"""


e = 2.7182818285


class Exponential:
    """Represents an exponential distribution.

    Attributes:
        lambtha (float): rate, the expected number of events per unit
            of time.
    """
    def __init__(self, data=None, lambtha=1.):
        """Set up the distribution from a parameter or from data.

        Args:
            data (list, optional): observed waiting times. If given,
                lambtha is estimated as 1 / sample mean and the lambtha
                argument is ignored.
            lambtha (float): rate of events. Used only when data is None.
                Must be positive.

        Raises:
            ValueError: if lambtha is not positive, or data has fewer
                than two values.
            TypeError: if data is not a list.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            # The mean waiting time is 1 / lambtha, so invert the sample mean
            mean = sum(data) / len(data)
            self.lambtha = 1 / mean

    def pdf(self, x):
        """Compute the probability density function.

        Args:
            x (float): waiting time.

        Returns:
            float: density at x, or 0 if x is negative.
        """
        if x < 0:
            return 0
        return self.lambtha * e ** (-self.lambtha * x)

    def cdf(self, x):
        """Compute the cumulative distribution function.

        Args:
            x (float): waiting time.

        Returns:
            float: probability that the wait is at most x
                (1 - e ** (-lambtha * x)), or 0 if x is negative.
        """
        if x < 0:
            return 0
        return 1 - e ** (-self.lambtha * x)
