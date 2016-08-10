#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reddit CNN.

This will read some posts from reddit database and train a convolutional
neural network on them with 1 convolutional layer using keras/theano.
So far, this is just using the imdb.py example from keras and inputting
the reddit data, but will update the model architecture to fit the data
better soon.

Usage:
    $ python reddit_cnn.py

TODO:
    * data cleaning, this is _actually_ important
    * num_filters, filter_length, hidden_dims
    * actually more than 1 subreddit?
    * init, activation
"""

from __future__ import print_function
import sqlite3
import re
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2
from keras.utils.visualize_util import plot

# ---------- Parameters ----------
np.random.seed(2222)

qry_lmt = 30000
vocab_size = 5000
embedding_dims = 100
paddedlength = 100   # length to which each sentence is padded
num_filters = 25     # number of filters to apply/learn in the 1D convolutional
filter_length = 2    # linear length of each filter (this is 1D)

# hidden_dimsa = 250 # number of output neurons for the first Dense layer
hidden_dimsb = 100   # number of output neurons for the second Dense layer

batch_size = 32      # TODO: which batch size is appropriate
epochs = 5           # number of training epochs

# ---------- Reading the reddit data ----------
# TODO: outsource to a nice wrapper
print("Querying db...")

# Read a list of subreddits that are supposed to be used from a file.
# This helps narrowing down the data to more frequent/appropriate subreddits.
with open('subreddits.txt', 'r') as f:
    l = f.read().splitlines()
    subreddits = ', '.join("'{0}'".format(s) for s in l)

sql_conn = sqlite3.connect("database.sqlite")

# TODO: randomize selection properly instead of taking the first x always
#       from the same subreddit
some_data = sql_conn.execute("SELECT subreddit, body, score FROM May2015\
                              WHERE subreddit IN (%s)\
                              LIMIT " % subreddits + str(qry_lmt))

print("Building corpus...")
raw_corpus, corpus, labels, strata = [], [], [], []

# TODO: insert proper data cleaning routine here
for post in some_data:
    raw_corpus.append(re.sub('\n', '', post[1]))
    cln_post = re.sub('[^A-Za-z0-9\.\,]+', ' ', post[1]).strip().lower()
    corpus.append(str(cln_post))
    labels.append(post[2])
    strata.append(post[0])

# ---------- Preparing the data matrices ----------
print("Creating train/test split")

tokenizer = Tokenizer(nb_words=vocab_size)
tokenizer.fit_on_texts(corpus)
seq = tokenizer.texts_to_sequences(corpus)

X = sequence.pad_sequences(seq, maxlen=paddedlength)

Y = np.asarray(labels)
Y = Y > 0
Y.astype(int)

# TODO: make a nicer and more coherent way to do the crossvalidation split {
X2 = X[25000:, ]
Y2 = Y[25000:, ]
X = X[:25000, ]
Y = Y[:25000, ]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# }

print(X_train.shape)

# ---------- Defining the model layers ----------
# TODO: how to initialize the weights?
nn = Sequential()

# Embedding
nn.add(Embedding(input_dim=vocab_size, output_dim=embedding_dims,
                 input_length=paddedlength))

# Convolution
nn.add(Convolution1D(
    num_filters,
    filter_length,
    activation="relu",
    W_regularizer=l2(3)
))
# TODO: check if l2-regularization is correct

nn.add(MaxPooling1D(pool_length=2))

nn.add(Flatten())

# nn.add(Dense(hidden_dimsa))
# nn.add(Dropout(0.25))
# nn.add(Activation('relu'))

nn.add(Dense(hidden_dimsb))
nn.add(Dropout(0.25))
nn.add(Activation('relu'))

nn.add(Dense(1))
nn.add(Activation('sigmoid'))

# TODO: check SGD update rule
nn.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

plot(nn, to_file='model.png')

# ---------- Fitting the model ----------
nn.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
       validation_data=(X_test, Y_test))
results = nn.evaluate(X_test, Y_test, verbose=1)

print(results)

# ---------- Crossvalidation ----------
# TODO: k-fold crossvalidation

print("Validating using secondary test set")
results2 = nn.evaluate(X2, Y2, verbose=1)
print(results2)
