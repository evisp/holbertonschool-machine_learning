#!/usr/bin/env python3
"""
Task 6: sort and tranpose a pandas dataframe
"""


def flip_switch(df):
    """ sort and tranpose a pandas dataframe """
    df = df.sort_index(ascending=False).T
    return df
