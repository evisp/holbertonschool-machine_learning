#!/usr/bin/env python3
"""Bidirectional Cell Forward Module"""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of a RNN:

    class constructor def __init__(self, i, h, o):
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by that
        represent the weights and biases of the cell
            Whf and bhfare for the hidden states in the forward direction
            Whb and bhbare for the hidden states in the backward direction
            Wy and byare for the outputs
        The weights should be initialized using a random normal distribution in
        the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros

    public instance method def forward(self, h_prev, x_t): that performs
    forward propagation for one time step"""

    def __init__(self, i, h, o):
        self.Whf = np.random.normal(size=(i+h, h))
        self.Whb = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h*2, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input for
        the cell
        m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state
        The output of the cell should use a softmax activation function

        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell"""

        h_x = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(h_x, self.Whf) + self.bhf)

        return h_next
