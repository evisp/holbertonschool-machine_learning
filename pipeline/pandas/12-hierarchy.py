#!/usr/bin/env python3
"""
Task 12
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    """
    df1 = index(df1)
    df2 = index(df2)
    df = pd.concat(
        [df2.loc[1417411980:1417417980], df1.loc[1417411980:1417417980]],
        keys=['bitstamp', 'coinbase']
    )
    df = df.swaplevel()
    df = df.sort_index(level='Timestamp')
    return df
