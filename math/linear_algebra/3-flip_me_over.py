#!/usr/bin/env python3
"""Write a function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """Transpose a given 2D matrix"""
    transpose = [[matrix[j][i] for j in range(len(matrix))]
                 for i in range(len(matrix[0]))]
    return transpose
