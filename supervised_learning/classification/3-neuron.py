#!/usr/bin/env python3
"""
    Class Neuron
"""

import numpy as np


class Neuron:
    """
        Class Neuron : define single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
            The weights vector for the neuron

            :return: value for private attribute __W
        """
        return self.__W

    @property
    def b(self):
        """
            The bias for the neuron

            :return: value for private attribute __b
        """
        return self.__b

    @property
    def A(self):
        """
            The activated output of the neuron (prediction)

            :return: value for private attribute __A
        """
        return self.__A

    def forward_prop(self, X):
        """
            method to calculate the forward propagation of the neuron

            :param X: ndarray (shape (nx, m)) contains input data

            :return: forward propagation
        """
        # multiplication of weight and add bias
        Z = np.matmul(self.__W, X) + self.__b

        # activation function
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
            Method to calculate cost (using logistic regression)

            :param Y: ndarray shape(1,m) correct labels
            :param A: ndarray shape(1,m) activated output

            :return: the cost
        """
        # store m value
        m = Y.shape[1]

        # calculate log loss function
        log_loss = -(1 / m) * np.sum((Y * np.log(A) + (1-Y) *
                                      np.log(1.0000001 - A)))

        return log_loss
