#!/usr/bin/env python3
"""DenseNet-121 Module"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds a Transition Layer.

    Args:
        growth_rate: the growth rate.

        compression: the compression factor.

    Returns:
        The keras model.
    """

    X = K.Input(shape=(224, 224, 3))
    nb_filters = 2 * growth_rate
    init = K.initializers.he_normal

    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.ReLU()(batch_norm)
    conv2d = K.layers.Conv2D(nb_filters, 7, strides=2,
                             padding="same",
                             kernel_initializer=init)(activation)
    max_pool = K.layers.MaxPool2D(3, 2, padding="same")(conv2d)

    dense_block_0, nb_filters = dense_block(
        max_pool, nb_filters, growth_rate, 6)
    transition_layer_0, nb_filters = transition_layer(
        dense_block_0, nb_filters, compression)

    dense_block_1, nb_filters = dense_block(
        transition_layer_0, nb_filters, growth_rate, 12)
    transition_layer_1, nb_filters = transition_layer(
        dense_block_1, nb_filters, compression)

    dense_block_2, nb_filters = dense_block(
        transition_layer_1, nb_filters, growth_rate, 24)
    transition_layer_2, nb_filters = transition_layer(
        dense_block_2, nb_filters, compression)

    dense_block_3, nb_filters = dense_block(
        transition_layer_2, nb_filters, growth_rate, 16)

    average_pool = K.layers.AveragePooling2D(
        7, padding="same")(dense_block_3)
    Y = K.layers.Dense(
        1000, "softmax", kernel_initializer=init)(average_pool)

    return K.Model(inputs=X, outputs=Y)
