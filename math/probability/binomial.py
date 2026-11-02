#!/usr/bin/env python3
"""Binomial distribution.

The binomial distribution models the number of successes in n independent
trials, each succeeding with the same probability p (heads in n coin
flips, clicks among n ad views, defects in a batch of n items).

It is described by n and p. Its probability mass function is:

    P(k) = n! / (k! * (n - k)!) * p ** k * (1 - p) ** (n - k)
"""


def factorial(n):
    """Return n! computed with a loop (1 for n < 2)."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


class Binomial:
    """Represents a binomial distribution.

    Attributes:
        n (int): number of trials.
        p (float): probability of success on each trial.
    """
    def __init__(self, data=None, n=1, p=0.5):
        """Set up the distribution from parameters or from data.

        Args:
            data (list, optional): observed success counts, one per
                experiment. If given, n and p are estimated from it and
                the other arguments are ignored.
            n (int): number of trials. Used only when data is None.
                Must be positive.
            p (float): probability of success. Used only when data is
                None. Must be strictly between 0 and 1.

        Raises:
            ValueError: if n is not positive, p is not in (0, 1), or data
                has fewer than two values.
            TypeError: if data is not a list.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            squared_deviations = 0
            for x in data:
                squared_deviations += (x - mean) ** 2
            variance = squared_deviations / len(data)
            # From mean = n*p and variance = n*p*(1-p):
            #   1 - p = variance / mean  =>  n = mean / (1 - variance / mean)
            # n must be an integer, so round it, then recompute p from it.
            self.n = round(mean / (1 - (variance / mean)))
            self.p = mean / self.n

    def pmf(self, k):
        """Compute the probability mass function.

        Args:
            k (int or float): number of successes. Non-integers are
                truncated to an integer.

        Returns:
            float: probability of exactly k successes, or 0 if k is
                negative.
        """
        k = int(k)
        if k < 0:
            return 0
        # Binomial coefficient: ways to choose which k trials succeed
        coefficient = factorial(self.n) / (
            factorial(k) * factorial(self.n - k))
        return coefficient * self.p ** k * (1 - self.p) ** (self.n - k)

    def cdf(self, k):
        """Compute the cumulative distribution function.

        Args:
            k (int or float): number of successes. Non-integers are
                truncated to an integer.

        Returns:
            float: probability of at most k successes, or 0 if k is
                negative.
        """
        k = int(k)
        if k < 0:
            return 0
        # Add up the probability of every outcome from 0 to k
        total = 0
        for i in range(k + 1):
            total += self.pmf(i)
        return total
