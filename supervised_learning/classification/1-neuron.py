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
