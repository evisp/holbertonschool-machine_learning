#!/usr/bin/env python3
"""
    Precision
"""

import numpy as np


def precision(confusion):
    """
        function that calculates the precision for each class
        in a confusion matrix

        :param confusion: ndarray, shape(classes,classes), confusion matrix

        :return: ndarray, shape(classes,), precision for each class
    """
    # number of classes
    classes = confusion.shape[0]
    # initialize sensitivity
    precision_matrix = np.zeros((classes,))

    for i in range(classes):
        true_positive = confusion[i, i]
        # Sum of all elements in column
        false_positive = np.sum(confusion[:, i]) - true_positive
        precision_matrix[i] = true_positive / (true_positive + false_positive)

    return precision_matrix
