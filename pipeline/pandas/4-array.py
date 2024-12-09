#!/usr/bin/env python3
"""
Task 4: convert pandas dataframe to numpy array
"""


def array(df):
    """ convert pandas dataframe to numpy array """
    df = df[['High', 'Close']].tail(10).to_numpy()
    return df