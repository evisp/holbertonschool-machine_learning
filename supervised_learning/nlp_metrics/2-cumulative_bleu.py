#!/usr/bin/env python3
"""Cumulative N-gram BLEU Score"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence:
        - references is a list of reference translations
        - each reference translation is a list of the words in the translation
        - sentence is a list containing the model proposed sentence
        - n is the size of the n-gram to use for evaluation
        - All n-gram scores should be weighted evenly

    Returns: the cumulative n-gram BLEU score
    """

    BP = min(1, np.exp(1 - len(min(references, key=len)) / len(sentence)))
    precision = []

    for m in range(1, n+1):
        n_grams = []
        for reference in references:
            n_grams_ref = []
            for i in range(len(sentence) - (m - 1)):
                if any(sentence[i:i + m] == reference[j:j+m]
                       for j in range(len(reference) - (m - 1))) and \
                        sentence[i:i+m] not in n_grams_ref:
                    n_grams_ref.append(sentence[i:i+m])
            n_grams.append(len(n_grams_ref))
        precision.append(max(n_grams) / (i + 1))

    return BP * np.exp(np.mean(np.log(precision)))
