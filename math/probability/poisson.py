#!/usr/bin/env python3
"""  a class Poisson that represents a poisson distribution """


class Poisson:
    def __init__(self, data=None, lambtha=1.):
        """ represents a poisson distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
    
    def pmf(self, k):
        """
        Calculates the PMF for a given number of “successes”.

        Args:
            k (int or float): The number of “successes”.

        Returns:
            float: The PMF value for k.
        """
        if not isinstance(k, int):
            try:
                k = int(k)
            except ValueError:
                return 0
        
        if k < 0:
            return 0

        # Compute PMF using the formula: P(k; λ) = (λ^k * e^(-λ)) / k!
        return (self.lambtha ** k * math.exp(-self.lambtha)) / math.factorial(k)
