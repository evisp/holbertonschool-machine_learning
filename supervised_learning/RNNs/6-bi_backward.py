#!/usr/bin/env python3
"""Bidirectional Cell Backward Module"""
import numpy as np


class BidirectionalCell:
    """
        Represents a bidirectional cell of a RNN
    """

    def __init__(self, i, h, o):

        self.Whf = np.random.normal(size=(i+h, h))
        self.Whb = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h*2, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
            Performs forward propagation for one time step
        """

        h_x = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(h_x, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """Calculates the hidden state in the backward direction """

        h_x = np.concatenate((h_next, x_t), axis=1)

        h_prev = np.tanh(np.dot(h_x, self.Whb) + self.bhb)

        return h_prev
