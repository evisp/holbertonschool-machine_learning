#!/usr/bin/env python3
"""
Down-sample a DataFrame by keeping only some rows and columns.
"""


def slice(df):
    """
    Select four price and volume columns, keeping every 60th row.

    Args:
        df: a DataFrame with the columns "High", "Low", "Close" and
            "Volume_(BTC)". Each row is one minute of data.

    Returns:
        A DataFrame with those four columns and one row per hour.
    """
    # ::60 takes every 60th row, which is one row per hour for minute data
    df = df.loc[::60, ['High', 'Low', 'Close', 'Volume_(BTC)']]
    return df
