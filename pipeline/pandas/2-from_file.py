#!/usr/bin/env python3
""" Defines function that loads data from a file as a Pandas DataFrame
"""


import pandas as pd


def from_file(filename, delimiter):
    """ Loads data from a file as a Pandas DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df