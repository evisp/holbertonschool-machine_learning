#!/usr/bin/env python3
"""
Extract the most recent prices from a DataFrame as a NumPy array.
"""


def array(df):
    """
    Return the last 10 High and Close values as a NumPy array.

    Args:
        df: a DataFrame with "High" and "Close" columns.

    Returns:
        A numpy.ndarray of shape (10, 2): the High and Close values of the
        last 10 rows.
    """
    # Select two columns, keep the last 10 rows, then convert to NumPy
    df = df[['High', 'Close']].tail(10).to_numpy()
    return df
