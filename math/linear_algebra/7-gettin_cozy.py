#!/usr/bin/env python3
""" defines function that concatenates two 2D matrices along an axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """Returns a new matrix that is the concatenation of two 2D matrices."""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        cat_matrix = []
        for row in mat1:
            cat_matrix.append(list(row))
        for row in mat2:
            cat_matrix.append(list(row))
        return cat_matrix
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        cat_matrix = [list(mat1_row + mat2_row) for mat1_row, mat2_row in zip(mat1, mat2)]
        return cat_matrix
    else:
        return None
