#!/usr/bin/env python3
"""Dense Block Module"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a Transition Layer.

    Args:
        X: the output from the previous layer.

        nb_filters: an integer representing the number of filters in X.

        compression: the compression factor for the transition layer.

    Returns:
        The output of the transition layer and the number of filters within
        the output, respectively.
    """

    init = K.initializers.he_normal
    compressed_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(compressed_filters, 1,
                        padding="same", kernel_initializer=init)(X)
    X = K.layers.AveragePooling2D(2, 2)(X)

    return X, compressed_filters
