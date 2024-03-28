#!/usr/bin/env python3
"""
DeepNeuralNetwork performing binary classification
"""


import numpy as np


class DeepNeuralNetwork:
    """
    Class that represents a deep neural network for binary classification
    """

    def __init__(self, nx, layers):
        """
        Class constructor
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.cache = {}
        self.weights = {}
        previous = nx

        for l, layer_size in enumerate(layers, 1):
            self.weights[f"W{l}"] = np.random.randn(layer_size, previous) * np.sqrt(2 / previous)
            self.weights[f"b{l}"] = np.zeros((layer_size, 1))
            previous = layer_size

        self.L = len(layers)