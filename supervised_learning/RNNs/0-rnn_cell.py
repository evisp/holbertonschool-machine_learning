#!/usr/bin/env python3
"""RNN Cell Module"""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN:

    class constructor def __init__(self, i, h, o):
    i is the dimensionality of the data
    h is the dimensionality of the hidden state
    o is the dimensionality of the outputs
    Creates the public instance attributes Wh, Wy, bh, by that represent the
    weights and biases of the cell
    Wh and bh are for the concatenated hidden state and input data
    Wy and by are for the output
    The weights should be initialized using a random normal distribution in the
    order listed above
    The weights will be used on the right side for matrix multiplication
    The biases should be initialized as zeros

    public instance method def forward(self, h_prev, x_t): that performs
    forward propagation for one time step"""

    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
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

        def softmax(x):
            """Compute softmax activation function"""
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        h_x = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(h_x, self.Wh) + self.bh)
        output_t = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, output_t
