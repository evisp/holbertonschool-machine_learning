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
        import math
        if not isinstance(k, int):
            k = int(k)
        
        if k < 0:
            return 0

        e = 2.7182818285

        # Compute PMF using the formula: P(k; λ) = (λ^k * e^(-λ)) / k!
        return (e**(-self.lambtha) * self.lambtha ** k ) / math.factorial(k)

    def cdf(self, k):
        """
        Calculates the CDF for a given number of “successes”.

        Args:
            k (int or float): The number of “successes”.

        Returns:
            float: The CDF value for k.
        """
        import math
        if not isinstance(k, int):
            try:
                k = int(k)
            except ValueError:
                return 0
        
        if k < 0:
            return 0

        # Compute CDF as the sum of PMF values from 0 to k
        cdf = sum(self.pmf(i) for i in range(k + 1))
        return cdf
