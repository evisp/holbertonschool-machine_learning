#!/usr/bin/env python3
"""
Task 100
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    """
    np.random.seed(5)
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    mountain = plt.scatter(x, y, c=z)
    plt.title('Mountain Elevation')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.colorbar(mountain, label='elevation (m)')
    plt.show()
