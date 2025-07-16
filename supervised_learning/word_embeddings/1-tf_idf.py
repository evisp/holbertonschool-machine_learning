#!/usr/bin/env python3
"""
NLP --WE --Task 1
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    E = vectorizer.fit_transform(sentences)
    F = vectorizer.get_feature_names_out()
    return E.toarray(), F
