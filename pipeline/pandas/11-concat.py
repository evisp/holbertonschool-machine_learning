#!/usr/bin/env python3
"""
Combine two exchanges' price data into a single DataFrame.
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Stack bitstamp data on top of coinbase data, labelling each source.

    Args:
        df1: a coinbase DataFrame with a "Timestamp" column.
        df2: a bitstamp DataFrame with a "Timestamp" column.

    Returns:
        A DataFrame indexed by Timestamp, with the exchange name as an extra
        outer index level. It holds the bitstamp rows up to and including
        Timestamp 1417411920, followed by all the coinbase rows. That is the
        minute before coinbase starts recording.
    """
    # Both frames are indexed by Timestamp so rows can be sliced by time
    df1 = index(df1)
    df2 = index(df2)

    # keys adds the outer level that tells the two sources apart
    df = pd.concat([df2.loc[:1417411920], df1], keys=['bitstamp', 'coinbase'])
    return df
