#!/usr/bin/env python3
"""
Task 4
"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    """
    inputs = K.Input(shape=(224, 224, 3))
    he_normal = K.initializers.he_normal(seed=0)
    X = K.layers.Conv2D(64,
                        (7, 7),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=he_normal)(inputs)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # X=K.layers.ReLU()(X)

    X = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    X = projection_block(X, [64, 64, 256], s=1)
    for _ in range(2):
        X = identity_block(X, [64, 64, 256])

    X = projection_block(X, [128, 128, 512])
    for _ in range(3):
        X = identity_block(X, [128, 128, 512])

    X = projection_block(X, [256, 256, 1024])
    for _ in range(5):
        X = identity_block(X, [256, 256, 1024])

    X = projection_block(X, [512, 512, 2048])
    for _ in range(2):
        X = identity_block(X, [512, 512, 2048])

    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(X)
    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=he_normal)(X)

    return K.models.Model(inputs=inputs, outputs=Y)
