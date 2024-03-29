#!/usr/bin/env python3
"""
DeepNeuralNetwork performing binary classification
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    class that represents a deep neural network
    """
    def __init__(self, nx, layers, activation='sig'):
        """
            class constructor

            :param nx: number of input features
            :param layers: number of nodes in each layer
            :param activation: type of activation in hidden layer

        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if (not isinstance(layers, list) or
                not all(map(lambda x: isinstance(x, int) and x > 0, layers))):
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        # private attribute
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        # initialize parameters with He method
        for i in range(self.__L):
            if i == 0:
                self.__weights["W" + str(i + 1)] = (np.random.randn(layers[i],
                                                                    nx)
                                                    * np.sqrt(2 / nx))
            else:
                self.__weights["W" + str(i + 1)] = \
                    (np.random.randn(layers[i],
                                     layers[i - 1])
                     * np.sqrt(2 / layers[i - 1]))
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    
    @property
    def L(self):
        """
        gets the private instance attribute __L
        """
        return (self.__L)

    @property
    def cache(self):
        """
        gets the private instance attribute __cache
        """
        return (self.__cache)

    @property
    def weights(self):
        """
        gets the private instance attribute __weights
        """
        return (self.__weights)

    def forward_prop(self, X):
        """
            method calculate forward propagation of neural network

            :param X: ndarray, shape(nx,m) input data

            :return: output neural network and cache
        """

        # store X in A0

        self.__cache['A0'] = X
        L = self.__L

        for l in range(1, L):
            Z = (np.matmul(self.__weights["W" + str(l)],
                           self.__cache['A' + str(l - 1)]) +
                 self.__weights['b' + str(l)])
            if self.__activation == 'sig':
                A = 1 / (1 + np.exp(-Z))
            else:
                A = np.tanh(Z)
            self.__cache['A' + str(l)] = A

        Z = (np.matmul(self.__weights["W" + str(L)],
                       self.__cache['A' + str(L - 1)]) +
             self.__weights['b' + str(L)])
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        self.__cache['A' + str(L)] = A

        return A, self.__cache

    def evaluate(self, X, Y):
        """
            Method to evaluate the network's prediction

            :param X: ndarray shape(nx,m) contains input data
            :param Y: ndarray shape (1,m) correct labels

            :return: network's prediction and cost of the network
        """

        # run forward propagation
        output, cache = self.forward_prop(X)

        # calculate cost
        cost = self.cost(Y, output)

        # convert predicted proba to one-hot
        result = np.zeros_like(output)

        # label values
        result[np.argmax(output, axis=0), np.arange(output.shape[1])] = 1

        return result, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Method calculate one pass of gradient descent
            on neural network

            :param Y: ndarray, shape(1,m), correct labels
            :param cache: dictionary containing all intermediary value of
             network
            :param alpha: learning rate

        """
        L = self.__L

        # store m
        m = Y.shape[1]

        # derivative of final layer (output=self.L)
        dZ = cache['A' + str(L)] - Y
        dW = np.matmul(dZ, cache['A' + str(L - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W_prev = np.copy(self.__weights['W' + str(L)])
        self.__weights['W' + str(L)] -= alpha * dW
        self.__weights['b' + str(L)] -= alpha * db

        for l in range(L - 1, 0, -1):
            dA = np.matmul(W_prev.T, dZ)
            A = cache['A' + str(l)]
            if self.__activation == 'sig':
                dZ = dA * A * (1 - A)
            else:
                dZ = dA * (1 - (A ** 2))
            dW = np.matmul(dZ, cache['A' + str(l - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W_prev = np.copy(self.__weights['W' + str(l)])
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        trains the neuron and updates __weights and __cache
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            import matplotlib.pyplot as plt
            x_points = np.arange(0, iterations + 1, step)
            points = []
        for itr in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and (itr % step) == 0:
                cost = self.cost(Y, A)
                print("Cost after " + str(itr) + " iterations: " + str(cost))
            if graph and (itr % step) == 0:
                cost = self.cost(Y, A)
                points.append(cost)
            self.gradient_descent(Y, cache, alpha)
        itr += 1
        if verbose:
            cost = self.cost(Y, A)
            print("Cost after " + str(itr) + " iterations: " + str(cost))
        if graph:
            cost = self.cost(Y, A)
            points.append(cost)
            y_points = np.asarray(points)
            plt.plot(x_points, y_points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return (self.evaluate(X, Y))

    def save(self, filename):
        """
        saves the instance object to a file in pickle format
        """
        if not isinstance(filename, str):
            print("Error: filename must be a string.")
            return

        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        loads a pickled DeepNeuralNetwork object from a file
        """
        import pickle
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                return obj
        except FileNotFoundError:
            return None
