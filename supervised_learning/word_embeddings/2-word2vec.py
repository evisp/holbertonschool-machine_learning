#!/usr/bin/env python3
"""
NLP --WE --Task 2
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    """
    # Set the training algorithm based on cbow parameter
    sg = 0 if cbow else 1

    # Create the Word2Vec model
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
        )

    # prepare the model vocabulary
    model.build_vocab(sentences)

    # Train the model
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
