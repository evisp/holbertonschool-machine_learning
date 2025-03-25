#!/usr/bin/env python3
"""
DeepNeuralNetwork performing binary classification
"""


import numpy as np
import pickle


class DeepNeuralNetwork:
    """
    Class that represents a deep neural network for binary classification
    """

    def __init__(self, nx, layers):
        """
        Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        weights = {}
        previous = nx

        for index, layer in enumerate(layers, 1):

            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")

            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        """
        gets the private instance attribute __L
        """
        return self.__L

    @property
    def cache(self):
        """
        gets the private instance attribute __cache
        """
        return self.__cache

    @property
    def weights(self):
        """
        gets the private instance attribute __weights
        """
        return self.__weights

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuronhe, respectively
        """
        self.__cache["A0"] = X

        for index in range(self.L):
            W = self.weights["W{}".format(index + 1)]
            b = self.weights["b{}".format(index + 1)]

            z = np.matmul(W, self.cache["A{}".format(index)]) + b
            A = 1 / (1 + (np.exp(-z)))

            self.__cache["A{}".format(index + 1)] = A

        return A, self.cache

    def cost(self, Y, A):
        """
            Calculate cost of the model using logistic regression

            :param Y: ndarray, shape(1,m) correct labels
            :param A: ndarray, shape(1,m) activated output

            :return: cost
        """

        # store m value
        m = Y.shape[1]

        # calculate log loss function
        log_loss = -(1 / m) * np.sum((Y * np.log(A) + (1 - Y) *
                                      np.log(1.0000001 - A)))

        return log_loss

    def evaluate(self, X, Y):
        """
        evaluates the neural network's predictions
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Method calculate one pass of gradient descent
            on neural network

            :param Y: ndarray, shape(1,m), correct labels
            :param cache: dictionary containing all intermediary value of
             network
            :param alpha: learning rate

        """

        # store m
        m = Y.shape[1]

        # derivative of final layer (output=self.L)
        dZ_f = cache["A{}".format(self.L)] - Y

        # back loop to calculate previous
        for layer in range(self.L, 0, -1):

            # activation previous layer
            A_p = cache["A{}".format(layer - 1)]

            # derivate
            dW = (1 / m) * np.matmul(dZ_f, A_p.T)
            db = (1 / m) * np.sum(dZ_f, axis=1, keepdims=True)

            # weight of current layer
            A = self.weights['W{}'.format(layer)]
            # derivate current layer
            dZ = np.matmul(A.T, dZ_f) * A_p * (1 - A_p)

            # update parameters W and b : new position
            self.__weights["W{}".format(layer)] -= alpha * dW
            self.__weights["b{}".format(layer)] -= alpha * db

            # update dz_f with new value found
            dZ_f = dZ

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
            Method to train deep neural network

            :param X: ndarray, shape(nx,m), input data
            :param Y: ndarray, shapte(1,m), correct labels
            :param iterations: number of iterations to train over
            :param alpha: learning rate
            :param verbose: boolean print or not information
            :param graph: boolean print or not graph
            :param step: int

            :return: evaluation of training after iterations
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # list to store cost /iter
        costs = []
        count = []

        for i in range(iterations + 1):
            # run forward propagation
            A, cache = self.forward_prop(X)

            # run gradient descent for all iterations except the last one
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            cost = self.cost(Y, A)

            # store cost for graph
            costs.append(cost)
            count.append(i)

            # verbose TRUE, every step + first and last iteration
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                # run evaluate
                print("Cost after {} iterations: {}".format(i, cost))

        # graph TRUE after training complete
        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Method to saves instance object to a file in pickle format

            :param filename: file which the object should be saved

        """
        # test extention
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        # open file in binary write mode
        with open(filename, 'wb') as file:
            # use pickel to dump the object into the file
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
            method to load a pickled DeepNeuralNetwork object

            :param filename: file from which object should be loaded

            :return: loaded object
                    or None if filename doesn't exist
        """

        try:
            # open file in binary mode
            with open(filename, 'rb') as file:
                # use pickle to load
                loaded_object = pickle.load(file)
            return loaded_object

        except FileNotFoundError:
            return None
