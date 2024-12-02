#!/usr/bin/env python3
""" a script that calculates the sum of squares"""


def summation_i_squared(n):
    """ a function that calculates the sum of squares"""
    if type(n) is not int or n < 1:
        return None
    result = (n * (n + 1) * (2 * n + 1)) / 6
    return int(result)
