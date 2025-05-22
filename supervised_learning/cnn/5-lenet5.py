#!/usr/bin/env python3
"""
Task 5
"""
from tensorflow import keras as K


def lenet5(X):
    """
    """
    he_normal = K.initializers.HeNormal(seed=0)
    A1 = K.layers.Conv2D(6, 5, activation='relu',
                         kernel_initializer=he_normal, padding='same')(X)
    A2 = K.layers.MaxPooling2D()(A1)
    A3 = K.layers.Conv2D(16, 5, activation='relu',
                         kernel_initializer=he_normal)(A2)
    A4 = K.layers.MaxPooling2D()(A3)
    A5 = K.layers.Flatten()(A4)
    A6 = K.layers.Dense(120, activation='relu',
                        kernel_initializer=he_normal)(A5)
    A7 = K.layers.Dense(84, activation='relu',
                        kernel_initializer=he_normal)(A6)
    Y = K.layers.Dense(10, activation='softmax',
                       kernel_initializer=he_normal)(A7)
    model = K.Model(inputs=X, outputs=Y)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
