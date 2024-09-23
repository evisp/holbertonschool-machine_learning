#!/usr/bin/env python3
"""PCA Module"""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset """

    # Step 1: Perform Singular Value Decomposition
    # U is an orthogonal matrix of shape (n, n).
    # S is a vector containing singular values
    # Vh is the transpose of V, containing the principal components.
    U, S, Vh = np.linalg.svd(X)

    V = Vh.T   # transposes Vh to get the matrix of principal components.

    cumsum = np.cumsum(S)   # calculates the cumulative sum of the singular values
    cumsum /= cumsum[-1]    # normalizes the cumulative sum to represent the fraction of total variance

    #  finds the index of the first component that allows the 
    #  cumulative variance to reach or exceed the specified threshold
    r = np.where(cumsum >= var)[0][0]

    # returns the matrix containing the selected principal components
    return V[:, :r + 1]
