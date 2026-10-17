#!/usr/bin/env python3
"""
Reverse the order of a DataFrame and swap its rows and columns.
"""


def flip_switch(df):
    """
    Sort the data in reverse chronological order, then transpose it.

    Args:
        df: a DataFrame whose index is in chronological order.

    Returns:
        The transposed DataFrame: the old rows are now columns, ordered from
        the latest to the earliest.
    """
    # Latest row first, then rows become columns
    df = df.sort_index(ascending=False).T
    return df
