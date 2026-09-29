#!/usr/bin/env python3
"""Write a function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if not matrix[0]:
            return shape
        matrix = matrix[0]
    return shape
