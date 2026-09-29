#!/usr/bin/env python3
"""Write a function that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two given matrices along a specific axis"""
    if axis == 0 and len(mat1[0]) != len(mat2[0]) \
            or axis == 1 and len(mat1) != len(mat2):
        return None

    mat = mat1 + mat2 if axis == 0 else \
        [mat1[i] + mat2[i] for i in range(len(mat1))]

    return mat
