#!/usr/bin/env python3
"""PCA 1 Module"""
import numpy as np


def pca(X, ndim):
    """ Performs PCA on a dataset """
    X_m = X - np.mean(X, axis=0)

    U, S, Vh = np.linalg.svd(X_m)

    W = Vh.T[:, :ndim]

    return X_m @ W