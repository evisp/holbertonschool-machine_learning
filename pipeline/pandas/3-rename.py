#!/usr/bin/env python3
"""
Prepare a price DataFrame by renaming and converting its time column.
"""
import pandas as pd


def rename(df):
    """
    Turn the Timestamp column into readable datetimes and keep two columns.

    Args:
        df: a DataFrame with a "Timestamp" column of Unix times in seconds
            and a "Close" column.

    Returns:
        A DataFrame with only the columns "Datetime" and "Close".
    """
    # Unix seconds -> pandas datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    # Rename the column to match what it now holds
    df = df.rename(columns={'Timestamp': 'Datetime'})
    # Keep only the two columns of interest
    df = df[['Datetime', 'Close']]
    return df
