#!/usr/bin/env python3
""" defines function that transposes a 2D matrix """


def matrix_transpose(matrix):
    """Returns the transpose of the given 2D matrix."""
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0

    transpose = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = matrix[i][j]

    return transpose
