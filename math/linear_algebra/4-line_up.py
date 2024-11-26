#!/usr/bin/env python3
""" A script that adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """ A function that adds two arrays element-wise """
    if len(arr1) != len(arr2):
        return None

    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
