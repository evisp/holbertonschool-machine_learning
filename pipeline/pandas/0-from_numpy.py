#!/usr/bin/env python3
"""
Create a pandas DataFrame from a NumPy array.
"""
import pandas as pd


def from_numpy(array):
    """
    Convert a NumPy array into a DataFrame with alphabetical column labels.

    Args:
        array: a 2D numpy.ndarray to convert.

    Returns:
        A pandas.DataFrame whose columns are labelled A, B, C, ... in
        alphabetical order, one label per column of the array. The array
        can have at most 26 columns.
    """
    # Alphabet to draw the column labels from
    columns = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # Number of columns in the array = number of labels needed
    c = array.shape[1]
    df = pd.DataFrame(array, columns=list(columns[:c]))
    return df
