#!/usr/bin/env python3
"""
Remove incomplete rows from a DataFrame.
"""


def prune(df):
    """
    Drop every row that has no value in the "Close" column.

    Args:
        df: a DataFrame with a "Close" column.

    Returns:
        The DataFrame without the rows where "Close" is NaN.
    """
    df = df.dropna(subset=['Close'])
    return df
