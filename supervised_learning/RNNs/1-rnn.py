#!/usr/bin/env python3
"""RNN Module"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN:

    rnn_cell is an instance of RNNCell that will be used for the forward
    propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
    h is the dimensionality of the hidden state

    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs"""

    H = np.zeros((X.shape[0] + 1, h_0.shape[0], h_0.shape[1]))
    H[0] = h_0

    Y = np.zeros((X.shape[0], X.shape[1], rnn_cell.by.shape[1]))

    for i, x_t in enumerate(X):
        H[i + 1], Y[i] = rnn_cell.forward(H[i], x_t)

    return H, Y
