#!/usr/bin/env python3
"""Unigram BLEU Score"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
        - references is a list of reference translations
        - each reference translation is a list of the words in the translation
        - sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score
    """

    BP = min(1, np.exp(1 - len(min(references, key=len)) / len(sentence)))

    precision = max([sum(match in reference for match in set(sentence))
                     for reference in references]) / len(sentence)

    return BP * np.exp(np.log(precision))
