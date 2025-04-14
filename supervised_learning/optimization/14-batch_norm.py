#!/usr/bin/env python3
"""
   Batch Normalization upgraded
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tensor): The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation (function): The activation function
            that should be used on the output of the layer.

    Returns:
        tensor: A tensor of the activated output for the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Base layer with Dense
    layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False)
    x = layer(prev)

    # Batch normalization
    gamma = tf.Variable(tf.ones((n,)))
    beta = tf.Variable(tf.zeros((n,)))
    mean, variance = tf.nn.moments(x, axes=[0])
    epsilon = 1e-7
    x_normalized = (x - mean) / tf.sqrt(variance + epsilon)
    x_scaled = gamma * x_normalized + beta

    # Activation
    activated_output = activation(x_scaled)

    return activated_output
