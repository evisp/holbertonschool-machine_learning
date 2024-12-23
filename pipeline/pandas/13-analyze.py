#!/usr/bin/env python3
"""
Task 13
"""


def analyze(df):
    """
    """
    df = df.drop(columns=['Timestamp']).describe()
    return df
