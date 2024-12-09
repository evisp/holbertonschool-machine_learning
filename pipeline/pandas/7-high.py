#!/usr/bin/env python3
"""
Task 7: Sorts pandas dataframe by a column
"""


def high(df):
    """ Sorts pandas dataframe by a column """
    df = df.sort_values(by='High', ascending=False)
    return df
