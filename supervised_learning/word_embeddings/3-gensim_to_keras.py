#!/usr/bin/env python3
"""
NLP --WE --Task 3
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    """
    keys = model.wv
    weights = keys.vectors

    return tf.keras.layers.Embedding(input_dim=weights.shape[0],
                                     output_dim=weights.shape[1],
                                     weights=[weights],
                                     trainable=True)
