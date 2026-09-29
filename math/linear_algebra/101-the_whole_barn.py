#!/usr/bin/env python3
"""Write a function that adds two matrices"""
matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices(mat1, mat2):
    """Adds two matrices"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)

    if shape1 != shape2:
        return None

    if not isinstance(mat1, list):
        return mat1 + mat2

    result = []
    for i in range(shape1[0]):
        sub_matrix1 = mat1[i]
        sub_matrix2 = mat2[i]
        result.append(add_matrices(sub_matrix1, sub_matrix2))

    return result
