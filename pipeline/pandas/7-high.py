#!/usr/bin/env python3
"""
Sort a DataFrame by its highest prices.
"""


def high(df):
    """
    Sort the rows by the "High" column, from highest to lowest.

    Args:
        df: a DataFrame with a "High" column.

    Returns:
        The DataFrame sorted in descending order of "High".
    """
    df = df.sort_values(by='High', ascending=False)
    return df
