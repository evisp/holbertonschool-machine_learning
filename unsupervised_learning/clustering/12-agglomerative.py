#!/usr/bin/env python3
"""Agglomerative Module"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ Performs agglomerative clustering on a dataset """
    dendrogram = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(dendrogram, color_threshold=dist)

    plt.show()

    return scipy.cluster.hierarchy.fcluster(dendrogram, dist,
                                            criterion='distance')