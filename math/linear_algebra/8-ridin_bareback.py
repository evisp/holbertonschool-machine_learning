#!/usr/bin/env python3
""" defines function that performs matrix multiplication """


def mat_mul(mat1, mat2):
    """Returns a new matrix that is the product of two 2D matrices."""
    mat1_rows, mat1_columns = len(mat1), len(mat1[0])
    mat2_rows, mat2_columns = len(mat2), len(mat2[0])

    if mat1_columns != mat2_rows:
        return None

    new_matrix = []
    for i in range(mat1_rows):
        new_matrix.append([])
        for j in range(mat2_columns):
            dot_product = sum(mat1[i][k] * mat2[k][j] for k in range(mat1_columns))
            new_matrix[i].append(dot_product)

    return new_matrix
