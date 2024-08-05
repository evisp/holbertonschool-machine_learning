#!/usr/bin/env python3
"""Word2Vec Module"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """ creates and trains a gensim word2vec model """

    return Word2Vec(sentences=sentences, size=size, min_count=min_count,
                    window=window, negative=negative, iter=iterations,
                    seed=seed, workers=workers, sg=not cbow)
