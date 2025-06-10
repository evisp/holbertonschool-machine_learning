#!/usr/bin/env python3
"""
Task 2
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    """
    F11, F3, F12 = filters
    he_norm = K.initializers.he_normal(seed=0)

    X = K.layers.Conv2D(F11, (1, 1), kernel_initializer=he_norm)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=he_norm)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), kernel_initializer=he_norm)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
