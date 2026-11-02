#!/usr/bin/env python3
"""Poisson distribution.

The Poisson distribution models how many times an event happens in a fixed
interval, when events occur independently at a constant average rate
(calls per hour, typos per page, requests per second).

It is described by a single parameter, lambtha, the expected number of
events per interval. Its probability mass function is:

    P(k) = (e ** -lambtha) * (lambtha ** k) / k!
"""


e = 2.7182818285


def factorial(n):
    """Return n! computed with a loop (1 for n < 2)."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


class Poisson:
    """Represents a Poisson distribution.

    Attributes:
        lambtha (float): expected number of events per interval.
    """
    def __init__(self, data=None, lambtha=1.):
        """Set up the distribution from a parameter or from data.

        Args:
            data (list, optional): observed event counts. If given,
                lambtha is estimated as the sample mean and the lambtha
                argument is ignored.
            lambtha (float): expected number of events per interval.
                Used only when data is None. Must be positive.

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
            # For a Poisson distribution the mean equals lambtha
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Compute the probability mass function.

        Args:
            k (int or float): number of events. Non-integers are
                truncated to an integer.

        Returns:
            float: probability of exactly k events, or 0 if k is negative.
        """
        k = int(k)
        if k < 0:
            return 0
        return (e ** -self.lambtha * self.lambtha ** k) / factorial(k)

    def cdf(self, k):
        """Compute the cumulative distribution function.

        Args:
            k (int or float): number of events. Non-integers are
                truncated to an integer.

        Returns:
            float: probability of at most k events, or 0 if k is negative.
        """
        k = int(k)
        if k < 0:
            return 0
        # Sum lambtha**i / i! for i = 0..k, then apply e ** -lambtha once
        total = 0
        for i in range(k + 1):
            total += self.lambtha ** i / factorial(i)
        return total * e ** -self.lambtha
