#!/usr/bin/env python3
"""
Task 7
"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    """
    he_normal = K.initializers.he_normal(seed=0)
    inputs = K.Input(shape=(224, 224, 3))

    nb_filters = 2 * growth_rate

    # 7x7 conv, stride 2
    X = K.layers.BatchNormalization(axis=3)(inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(nb_filters, (7, 7), strides=2,
                        padding='same',
                        kernel_initializer=he_normal)(X)

    # 3x3 max pool, stride 2
    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)

    # Dense Block, 6 layers
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)

    # Transition
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block, 12 layers
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block, 24 layers
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block, 16 layers
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # 7x7 global average pool
    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(X)
    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=he_normal)(X)

    return K.models.Model(inputs=inputs, outputs=Y)
