#!/usr/bin/env python3
"""Write a function that performs element-wise addition, subtraction,
multiplication, and division"""


def np_elementwise(mat1, mat2):
    """ Performs element-wise addition, subtraction, multiplication,
    and division between two given matrices"""
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 * 1 / mat2
