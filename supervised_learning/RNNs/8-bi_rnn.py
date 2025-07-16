#!/usr/bin/env python3
"""Bidirectional RNN Module"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a simple RNN:

        bi_cell is an instance of BiDirectionalCell that will be used for the
        forward propagation
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state, given as a numpy.ndarray of
        shape (m, h)
            h is the dimensionality of the hidden state

        Returns: H, Y
            H is a numpy.ndarray containing all of the hidden states
            Y is a numpy.ndarray containing all of the outputs"""

    Hf = np.zeros((X.shape[0], h_0.shape[0], h_0.shape[1]))
    Hb = np.zeros(Hf.shape)
    Hf[0] = bi_cell.forward(h_0, X[0])
    Hb[-1] = bi_cell.backward(h_t, X[-1])

    for i in range(1, len(X)):
        x_tf = X[i]
        x_tb = X[-(i + 1)]

        Hf[i] = bi_cell.forward(Hf[i - 1], x_tf)
        Hb[-(i + 1)] = bi_cell.backward(Hb[-i], x_tb)

    H = np.concatenate((Hf, Hb), axis=-1)

    return H, bi_cell.output(H)
