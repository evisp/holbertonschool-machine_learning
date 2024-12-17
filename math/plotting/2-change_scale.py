#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    # your code here
    # plt.xlabel('Time (years)')
    # plt.ylabel('Fraction Remaining')
    # plt.title("Exponential Decay of C-14")
    # plt.yscale("log")
    # plt.xlim(0, 28650)
    # plt.plot(x, y)
    # plt.show()
    plt.plot(x, y)
    plt.yscale('log')
    plt.xlim([x[0], x[-1]])
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")
    plt.show()
