#!/usr/bin/env python3
"""Bag of Words Module"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """ create a bag of words embedding matrix """
    if vocab is None:
        vectorizer = CountVectorizer()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
