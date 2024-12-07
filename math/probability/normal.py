#!/usr/bin/env python3
"""Normal Class"""


pi = 3.1415926536
e = 2.7182818285


class Normal:
    """Class Normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Class constructor"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = 0
            for x in data:
                self.stddev += (x - self.mean) ** 2
            self.stddev = (self.stddev / len(data)) ** (1/2)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the PDF"""
        return (1 / (self.stddev * (2 * pi) ** (1 / 2))) * e ** (-(1 / 2) * ((
            x - self.mean) / self.stddev) ** 2)

    def cdf(self, x):
        """Calculates the CDF"""
        y = (x - self.mean) / (self.stddev * 2 ** (1 / 2))
        return 1 / 2 * (1 + (2 / pi ** (1 / 2) * (y - (y ** 3 / 3) + (
            y ** 5 / 10) - (y ** 7 / 42) + (y ** 9 / 216))))
