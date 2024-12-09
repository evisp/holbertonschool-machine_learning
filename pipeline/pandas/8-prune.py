#!/usr/bin/env python3
"""
Task 8: Removes any entries where Close has NaN values.
"""


def prune(df):
    """ Removes any entries where Close has NaN values"""
    df = df.dropna(subset=['Close'])
    return df
