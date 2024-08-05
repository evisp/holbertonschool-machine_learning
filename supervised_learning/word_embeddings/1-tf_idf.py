#!/usr/bin/env python3
"""TF-IDF Module"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ Creates a TF-IDF embedding """

    tfids_vector = TfidfVectorizer(vocabulary=vocab)
    embed = tfids_vector.fit_transform(sentences)
    features = tfids_vector.get_feature_names()
    return embed.toarray(), features
