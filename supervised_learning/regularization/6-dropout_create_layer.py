#!/usr/bin/env python3
"""
Task 6 : dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """

    """

    # Initialize layer weights using He initialization
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    # Create dense layer
    dense_layer = tf.keras.layers.Dense(units=n, activation=activation,
                                        kernel_initializer=init)

    # Apply dense layer to previous layer
    output = dense_layer(prev)

    # Apply dropout regularization
    dropout_layer = tf.keras.layers.Dropout(rate=1-keep_prob)
    output = dropout_layer(output, training=training)

    return output
