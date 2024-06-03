#!/usr/bin/env python3
"""
Defines a function that builds a projection block using Keras
"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block using Keras

    """

    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    activation = K.activations.relu

    C11 = K.layers.Conv2D(filters=F11,
                          kernel_size=(1, 1),
                          padding='same',
                          strides=s,
                          kernel_initializer=init)(A_prev)

    Batch_Norm11 = K.layers.BatchNormalization(axis=3)(C11)
    ReLU11 = K.layers.Activation(activation)(Batch_Norm11)

    C3 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=init)(ReLU11)

    Batch_Norm3 = K.layers.BatchNormalization(axis=3)(C3)
    ReLU3 = K.layers.Activation(activation)(Batch_Norm3)

    C12 = K.layers.Conv2D(filters=F12,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=init)(ReLU3)

    Batch_Norm12 = K.layers.BatchNormalization(axis=3)(C12)

    SC = K.layers.Conv2D(filters=F12,
                         kernel_size=(1, 1),
                         padding='same',
                         strides=s,
                         kernel_initializer=init)(A_prev)

    Batch_NormSC = K.layers.BatchNormalization(axis=3)(SC)

    Addition = K.layers.Add()([Batch_Norm12, Batch_NormSC])

    output = K.layers.Activation(activation)(Addition)

    return output
