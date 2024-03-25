#!/usr/bin/env python3
"""
Neuron class that defines a single neuron performing binary classification
"""


import numpy as np


class Neuron:
    """
    A single neuron performing binary classification
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
        get method for the private instance attribute __W
        """
        return (self.__W)

    @property
    def b(self):
        """
        get method for the private instance attribute __b
        """
        return (self.__b)

    @property
    def A(self):
        """
        get method for the private instance attribute _A
        """
        return (self.__A)
