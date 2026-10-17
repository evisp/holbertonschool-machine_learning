#!/usr/bin/env python3
"""
Compute descriptive statistics for a DataFrame.
"""


def analyze(df):
    """
    Summarise every column with count, mean, std, min, quartiles and max.

    Args:
        df: a DataFrame with a "Timestamp" column and numeric columns.

    Returns:
        A DataFrame of statistics, without the "Timestamp" column since
        statistics of a time label carry no meaning.
    """
    df = df.drop(columns=['Timestamp']).describe()
    return df
