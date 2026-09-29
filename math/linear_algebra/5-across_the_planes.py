#!/usr/bin/env python3
"""Write a function that adds two matrices element-wise"""
matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """Adds two given matrices element-wise"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    arr = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
           for i in range(len(mat1))]
    return arr
