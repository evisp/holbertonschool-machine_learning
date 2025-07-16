#!/usr/bin/env python3
"""LSTM Cell Module"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit:

    class constructor def __init__(self, i, h, o):
    i is the dimensionality of the data
    h is the dimensionality of the hidden state
    o is the dimensionality of the outputs
    Creates the public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo,
    by that represent the weights and biases of the cell
    Wf and bf are for the forget gate
    Wu and bu are for the update gate
    Wc and bc are for the intermediate cell state
    Wo and bo are for the output gate
    Wy and by are for the outputs
    The weights should be initialized using a random normal distribution in the
    order listed above
    The weights will be used on the right side for matrix multiplication
    The biases should be initialized as zeros

    public instance method def forward(self, h_prev, x_t): that performs
    forward propagation for one time step"""

    def __init__(self, i, h, o):
        self.Wf = np.random.normal(size=(i+h, h))
        self.Wu = np.random.normal(size=(i+h, h))
        self.Wc = np.random.normal(size=(i+h, h))
        self.Wo = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input for
        the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing the previous cell
        state
        The output of the cell should use a softmax activation function

        Returns: h_next, c_next, y
        h_next is the next hidden state
        c_next is the next cell state
        y is the output of the cell"""

        def softmax(x):
            """Compute softmax activation function"""
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        def sigmoid(x):
            """Compute sigmoid activation function"""
            return 1 / (1 + np.exp(-x))

        h_x = np.concatenate((h_prev, x_t), axis=1)

        f_t = sigmoid(np.dot(h_x, self.Wf) + self.bf)

        u_t = sigmoid(np.dot(h_x, self.Wu) + self.bu)
        c_tilde = np.tanh(np.dot(h_x, self.Wc) + self.bc)

        c_next = f_t * c_prev + u_t * c_tilde

        output_t = sigmoid(np.dot(h_x, self.Wo) + self.bo)
        h_next = output_t * np.tanh(c_next)

        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
