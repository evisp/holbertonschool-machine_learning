#!/usr/bin/env python3
"""
Task 3: rename a pandas dataframe
"""
import pandas as pd


def rename(df):
    """ rename a pandas dataframe """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df = df[['Datetime', 'Close']]
    return df
