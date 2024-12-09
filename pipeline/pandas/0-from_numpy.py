#!/usr/bin/env python3
""" create a pd.DataFrame from a np.ndarray """
import pandas as pd


def from_numpy(array):
    """ create a pd.DataFrame from a np.ndarray """
    columns = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    c = array.shape[1]
    df = pd.DataFrame(array, columns=list(columns[:c]))
    return df

