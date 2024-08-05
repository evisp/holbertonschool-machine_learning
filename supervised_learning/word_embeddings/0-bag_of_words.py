#!/usr/bin/env python3
"""Bag of Words Module"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix:

    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used

    Returns: embeddings, features
    embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
    s is the number of sentences in sentences
    f is the number of features analyzed
    features is a list of the features used for embeddings"""

    if vocab is None:
        vocab = []
        for sentence in sentences:
            vocab.extend(re.sub(r"\b\w{1}\b", "", re.sub(
                r"[^a-zA-Z0-9\s]", " ", sentence.lower())).split())
        vocab = sorted(list(set(vocab)))

    embeddings = np.zeros((len(sentences), len(vocab)))

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            word = re.sub(r"\b\w{1}\b", "", re.sub(
                r"[^a-zA-Z0-9\s]", " ", word.lower())).strip()
            if word in vocab:
                embeddings[i][vocab.index(word)] += 1

    return embeddings.astype(int), vocab
