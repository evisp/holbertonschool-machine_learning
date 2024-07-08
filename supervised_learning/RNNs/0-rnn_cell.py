#!/usr/bin/env python3
"""RNN Cell Module"""
import numpy as np


class RNNCell:
    """
        Represents a cell of a simple RNN:
    """

    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
            Performs forward propagation for one time step
        """

        def softmax(x):
            """Compute softmax activation function"""
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        h_x = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(h_x, self.Wh) + self.bh)
        output_t = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, output_t
