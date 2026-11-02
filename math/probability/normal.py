#!/usr/bin/env python3
"""Normal (Gaussian) distribution.

The normal distribution is the bell curve. It describes quantities that
cluster around an average with symmetric spread (heights, measurement
errors, many sums of small effects), and it underlies much of statistics
and machine learning.

It is described by its mean and standard deviation. Its probability density
function is:

    f(x) = 1 / (stddev * sqrt(2 * pi))
           * e ** (-0.5 * ((x - mean) / stddev) ** 2)
"""


pi = 3.1415926536
e = 2.7182818285


class Normal:
    """Represents a normal distribution.

    Attributes:
        mean (float): centre of the distribution.
        stddev (float): standard deviation, the spread around the mean.
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """Set up the distribution from parameters or from data.

        Args:
            data (list, optional): observed values. If given, mean and
                stddev are estimated from it and the other arguments are
                ignored.
            mean (float): mean of the distribution. Used only when data
                is None.
            stddev (float): standard deviation. Used only when data is
                None. Must be positive.

        Raises:
            ValueError: if stddev is not positive, or data has fewer
                than two values.
            TypeError: if data is not a list.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            # Population standard deviation: sqrt of the mean squared deviation
            squared_deviations = 0
            for x in data:
                squared_deviations += (x - self.mean) ** 2
            self.stddev = (squared_deviations / len(data)) ** (1 / 2)

    def z_score(self, x):
        """Convert a value to its z-score.

        Args:
            x (float): value on the original scale.

        Returns:
            float: number of standard deviations x lies from the mean.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Convert a z-score back to a value (inverse of z_score).

        Args:
            z (float): number of standard deviations from the mean.

        Returns:
            float: the corresponding value on the original scale.
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Compute the probability density function.

        Args:
            x (float): value at which to evaluate the density.

        Returns:
            float: density at x.
        """
        coefficient = 1 / (self.stddev * (2 * pi) ** (1 / 2))
        exponent = -(1 / 2) * self.z_score(x) ** 2
        return coefficient * e ** exponent

    def cdf(self, x):
        """Compute the cumulative distribution function.

        The CDF is 1/2 * (1 + erf((x - mean) / (stddev * sqrt(2)))). The
        error function erf has no closed form, so it is approximated here
        by the first five terms of its Maclaurin series. The approximation
        is accurate near the mean and loses accuracy far in the tails.

        Args:
            x (float): value at which to evaluate the CDF.

        Returns:
            float: probability that a value is at most x.
        """
        y = (x - self.mean) / (self.stddev * 2 ** (1 / 2))
        erf = 2 / pi ** (1 / 2) * (
            y - y ** 3 / 3 + y ** 5 / 10 - y ** 7 / 42 + y ** 9 / 216)
        return 1 / 2 * (1 + erf)
