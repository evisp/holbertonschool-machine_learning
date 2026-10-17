#!/usr/bin/env python3
"""
Build a multi-level index to compare two exchanges minute by minute.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Show bitstamp and coinbase rows side by side for a time window.

    Args:
        df1: a coinbase DataFrame with a "Timestamp" column.
        df2: a bitstamp DataFrame with a "Timestamp" column.

    Returns:
        A DataFrame with a two-level index (Timestamp, exchange name),
        sorted by Timestamp. It covers Timestamps 1417411980 to 1417417980
        inclusive, which is 100 minutes of data from each exchange.
    """
    df1 = index(df1)
    df2 = index(df2)
    # Keep only the shared time window, labelling each source
    df = pd.concat(
        [df2.loc[1417411980:1417417980], df1.loc[1417411980:1417417980]],
        keys=['bitstamp', 'coinbase']
    )
    # Put Timestamp first, then the exchange
    df = df.swaplevel()
    # Group the rows of both exchanges by minute
    df = df.sort_index(level='Timestamp')
    return df
