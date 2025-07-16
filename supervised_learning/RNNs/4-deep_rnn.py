#!/usr/bin/env python3
"""Deep RNN Module"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a simple RNN:

    rnn_cells is a list of RNNCell instances of length l that will be used for
    the forward propagation
        l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray
    of shape (l, m, h)
        h is the dimensionality of the hidden state

    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs"""

    H = np.zeros((X.shape[0] + 1, h_0.shape[0],
                  h_0.shape[1], h_0.shape[2]))
    H[0] = h_0

    Y = np.zeros((X.shape[0], X.shape[1], rnn_cells[-1].by.shape[1]))

    for i, x_t in enumerate(X):
        for l, rnn_cell in enumerate(rnn_cells):
            H[i + 1, l], Y[i] = rnn_cell.forward(H[i, l], x_t)
            x_t = H[i + 1, l]

    return H, Y
