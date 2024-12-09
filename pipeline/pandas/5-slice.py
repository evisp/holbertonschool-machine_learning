#!/usr/bin/env python3
"""
Task 5: slice a pandas dataframe
"""


def slice(df):
    """ slice a pandas dataframe """
    df = df.loc[::60, ['High', 'Low', 'Close', 'Volume_(BTC)']]
    return df
