#!/usr/bin/env python3
"""
Task 11
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Task 11
    """
    df1 = index(df1)
    df2 = index(df2)

    df = pd.concat([df2.loc[:1417411920], df1], keys=['bitstamp', 'coinbase'])
    return df
