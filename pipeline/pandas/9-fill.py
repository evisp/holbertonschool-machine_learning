#!/usr/bin/env python3
"""
Task 9: fill nan values in a pandas data frame
"""


def fill(df):
    """ fill nan values in a pandas data frame """
    df = df.drop(columns='Weighted_Price')
    df['Close'] = df['Close'].ffill(axis=0)

    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
