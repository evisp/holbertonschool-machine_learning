#!/usr/bin/env python3
"""Kmeans Sklearn Module"""
import sklearn.cluster


def kmeans(X, k):
    """ Performs K-means on a dataset """
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)

    return kmeans.cluster_centers_, kmeans.labels_
