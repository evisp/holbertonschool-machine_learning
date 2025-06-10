#!/usr/bin/env python3
"""
Task 3
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    """
    F11, F3, F12 = filters
    he_norm = K.initializers.he_normal(seed=0)

    X = K.layers.Conv2D(F11, (1, 1), strides=(s, s),
                        padding='valid', kernel_initializer=he_norm)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=he_norm)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), kernel_initializer=he_norm)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X_short = K.layers.Conv2D(F12, (1, 1), strides=(s, s),
                              padding='valid',
                              kernel_initializer=he_norm)(A_prev)
    X_short = K.layers.BatchNormalization(axis=3)(X_short)

    X = K.layers.Add()([X, X_short])
    X = K.layers.Activation('relu')(X)

    return X
