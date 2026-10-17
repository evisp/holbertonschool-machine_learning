#!/usr/bin/env python3
"""
Use the time column as the index of a DataFrame.
"""


def index(df):
    """
    Set the "Timestamp" column as the index of the DataFrame.

    Args:
        df: a DataFrame with a "Timestamp" column.

    Returns:
        The DataFrame indexed by "Timestamp".
    """
    return df.set_index('Timestamp')
