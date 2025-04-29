#!/usr/bin/env python3
"""
Task 2
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    """
    regularization_loss = model.losses
    return cost + regularization_loss
