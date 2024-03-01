#!/usr/bin/env python3
"""
Defines function that creates a Pandas DataFrame from a Numpy ndarray
"""


import pandas as pd
import string


def from_numpy(array):
    """
    Creates a Pandas DataFrame from a numpy.ndarray
    """
    alphabet = list(string.ascii_uppercase)
    column_labels = [alphabet[i] for i in range(len(array[0]))]
    df = pd.DataFrame(array, columns=column_labels)
    return df
