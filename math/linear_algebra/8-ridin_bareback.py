#!/usr/bin/env python3
"""Write a function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication with two given matrices"""
    if len(mat1[0]) != len(mat2):
        return None

    mat = [[sum(mat1[z][j] * mat2[j][i] for j in range(len(mat1[0])))
            for i in range(len(mat2[0]))] for z in range(len(mat1))]

    return mat
