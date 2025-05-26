#!/usr/bin/env python3
"""
Task 4
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    """
    return tf.image.random_brightness(image, max_delta)
