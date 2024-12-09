#!/usr/bin/env python3
"""
Task 2: loads data from a file as a pd.DataFrame
"""
import pandas as pd


def from_file(filename, delimiter):
    """ loads data from a file as a pd.DataFrame """
    return pd.read_csv(filename, delimiter=delimiter)
