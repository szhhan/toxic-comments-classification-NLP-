#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:40:46 2019

@author: sizhenhan
"""

from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
import numpy as np 
from tqdm import tqdm 

PORTER_STEMMER = PorterStemmer()
LANCASTER_STEMMER = LancasterStemmer()
SNOWBALL_STEMMER = SnowballStemmer("english")

def word_forms(word):
    yield word
    yield word.lower()
    yield word.upper()
    yield word.capitalize()
    yield PORTER_STEMMER.stem(word)
    yield LANCASTER_STEMMER.stem(word)
    yield SNOWBALL_STEMMER.stem(word)
    
    
def maybe_get_embedding(word, model):
    for form in word_forms(word):
        if form in model:
            return model[form]

    word = word.strip("-'")
    for form in word_forms(word):
        if form in model:
            return model[form]

    return None

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

def gensim_to_embedding_matrix(word2index, path):
    model = load_embeddings(path)
    embedding_matrix = np.zeros((max(word2index.values()) + 1, 300), dtype=np.float32)
    unknown_words = []

    for word, i in word2index.items():
        maybe_embedding = maybe_get_embedding(word, model)
        if maybe_embedding is not None:
            embedding_matrix[i] = maybe_embedding
        else:
            unknown_words.append(word)

    return embedding_matrix, unknown_words