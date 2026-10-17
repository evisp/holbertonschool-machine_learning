#!/usr/bin/env python3
"""
Load a delimited text file into a pandas DataFrame.
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Read a delimited text file, such as a CSV, into a DataFrame.

    Args:
        filename: path of the file to load.
        delimiter: the character that separates the columns in the file.

    Returns:
        A pandas.DataFrame with the contents of the file.
    """
    return pd.read_csv(filename, delimiter=delimiter)
