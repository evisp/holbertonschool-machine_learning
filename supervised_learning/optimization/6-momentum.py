#!/usr/bin/env python3
"""
   Momentum
"""


import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.

    Returns:
        optimizer: Optimizer object for gradient descent with momentum.
    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1)
    return optimizer
