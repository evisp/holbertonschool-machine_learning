#!/usr/bin/env python3
"""
   Learning Rate decay upgraded
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation
    in TensorFlow using inverse time decay.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight
            used to determine the rate at which alpha will decay.
        decay_step (int): The number of passes of
            gradient descent that should occur before alpha is decayed further.

    Returns:
        learning_rate_decay_op: The learning rate decay operation.
    """
    learning_rate_decay_op = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_rate=decay_rate,
        decay_steps=decay_step,
        staircase=True
    )
    return learning_rate_decay_op
