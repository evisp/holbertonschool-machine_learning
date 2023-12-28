#!/usr/bin/env python3
""" defines function that transposes a 2D matrix """


def matrix_transpose(matrix):
    """Returns the transpose of the given 2D matrix."""
    matrix_transpose = []

    for row in matrix:
        for idx, value in enumerate(row):
            if len(matrix_transpose) <= idx:
                matrix_transpose.append([])
            matrix_transpose[idx].append(value)

    return matrix_transpose
