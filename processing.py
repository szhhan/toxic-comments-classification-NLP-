#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:16:24 2019

@author: sizhenhan
"""
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import TweetTokenizer
from keras.preprocessing import text, sequence


def clean(x, data):
    y = np.where(data['target'] >= 0.5, 1, 0)
    y_aux_train = data[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
    yy = np.hstack([y[:, np.newaxis], y_aux_train])
    
    X_train, X_test, y_train, y_test = train_test_split(x, yy, test_size=0.3, random_state=2019)
    
    return X_train, X_test, y_train, y_test

def token(X_train, X_test):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    x_train = []
    word_dict = {}
    word_index = 1

    for doc in X_train:
        word_seq = []
        for word in tknzr.tokenize(doc):
            if word not in word_dict:
                word_dict[word] = word_index
                word_index += 1
            word_seq.append(word_dict[word])
        x_train.append(word_seq)
        
    x_train = sequence.pad_sequences(x_train, maxlen=200,padding = 'post')
    word_dict['unknown-words-in-test'] = 0 
    
    x_test = []
    for doc in X_test:
        word_seq = []
        for word in tknzr.tokenize(doc):
            if word in word_dict:
                word_seq.append(word_dict[word])
            else:
                word_seq.append(0)
        x_test.append(word_seq)
    
    x_test = sequence.pad_sequences(x_test, maxlen=200,padding = 'post')
    
    return x_train, x_test, word_dict 