#!/usr/bin/env python3
"""GRU Cell Module"""
import numpy as np


class GRUCell:
    """
        Represents a gated recurrent unit
    """

    def __init__(self, i, h, o):

        self.Wz = np.random.normal(size=(i+h, h))
        self.Wr = np.random.normal(size=(i+h, h))
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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

        r_t = 1 / (1 + np.exp(-(np.dot(h_x, self.Wr) + self.br)))

        z_t = 1 / (1 + np.exp(-(np.dot(h_x, self.Wz) + self.bz)))

        rh_x = np.concatenate((r_t * h_prev, x_t), axis=1)

        h_tilde = np.tanh(np.dot(rh_x, self.Wh) + self.bh)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        output_t = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, output_t
