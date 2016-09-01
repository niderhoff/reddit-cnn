#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reddit CNN - binary classification on reddit comment scores.

Reads posts and scores from reddit comment database, provide sequence
embeddings for the comment text to feed into various machine learning models.

Provides functions to train logistic regression from scikit-learn (simple) as
well as keras (simple and with l1 and l2 regularization).

Also trains Convolutional Neural Networks (CNN) with varying filter sizes,
filter numbers and optimizers to find optimal network for the data (still in
progress) using keras (theano).

Usage:
    $ python reddit_cnn.py

TODO:
    *  data cleaning, this is _actually_ important
    *  actually more than 1 subreddit?
    *  init, activation
    *  k-fold crossvalidation
"""

from __future__ import print_function

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
# from keras.utils.np_utils import accuracy
from keras.optimizers import SGD
from keras.regularizers import l1l2
# from keras.utils.visualize_util import plot

import preprocess as pre


def get_sequences(corpus, max_features=5000, maxlen=100):
    # Read a list of subreddits that are supposed to be used from a file.
    # This helps narrowing down the data to more frequent/appropriate
    # subreddits.
    tokenized = Tokenizer(nb_words=max_features)
    tokenized.fit_on_texts(corpus)
    seq = tokenized.texts_to_sequences(corpus)
    X = sequence.pad_sequences(seq, maxlen)
    return X


def get_labels_binary(labels, threshold=1):
    Y = np.asarray(labels)
    print('Y.shape: ' + str(Y.shape))
    print('Y: ' + str(Y))

    Ybool = abs(Y) > threshold
    print("Y > 1: " + str(Ybool))

    Ybin = Ybool.astype(int)
    print("Y binary: " + str(Ybin))

    return np.expand_dims(Ybin, axis=1)


def cnn_build(max_features, maxlen, embedding_dim, filter_size, nb_filter,
              dropout_p, activation="relu"):
    nn = Sequential()
    nn.add(Embedding(input_dim=max_features, output_dim=embedding_dim,
                     input_length=maxlen))
    nn.add(Dropout(dropout_p))
    nn.add(Convolution1D(
        nb_filter,
        filter_size,
        activation=activation
    ))
    nn.add(MaxPooling1D(pool_length=maxlen - filter_size + 1))
    nn.add(Flatten())
    nn.add(Dropout(dropout_p))
    nn.add(Dense(1))
    nn.add(Activation('sigmoid'))
    print(nn.summary())
    return nn


def cnn_train(model, X_train, y_train, validation_data=None, val=False,
              batch_size=32, nb_epoch=5, opt=SGD()):
    if (val is True and validation_data is not None):
        model.compile(loss='binary_crossentropy', optimizer=opt,
                      metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  validation_data=validation_data)
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt)
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)


def lr_train(X_train, y_train, val=True, validation_data=None, type='skl',
             nb_epoch=10, reg=l1l2(l1=0.01, l2=0.01)):
    X_test, y_test = validation_data
    if (type == 'skl'):
        lr = LogisticRegressionCV()
        lr.fit(X_train, y_train.ravel())
        pred_y = lr.predict(X_test)
        print("Test fraction correct (LR-Accuracy) = {:.6f}".format(lr.score(
              X_test, y_test)))
        return pred_y
    elif (type == 'k1'):
        # 2-class logistic regression in Keras
        model = Sequential()
        model.add(Dense(1, activation='sigmoid', input_dim=X_train.shape[1]))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, nb_epoch=nb_epoch,
                  validation_data=validation_data)
        return model.evaluate(X_test, y_test, verbose=0)
    elif (type == 'k2'):
        # logistic regression with L1 and L2 regularization
        model = Sequential()
        model.add(Dense(1, activation='sigmoid', W_regularizer=reg,
                  input_dim=X_train.shape[1]))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, nb_epoch=nb_epoch,
                  validation_data=validation_data)
        return model.evaluate(X_test, y_test, verbose=0)


# ---------- Data gathering ----------
qry_lmt = 100000
# subreddits = subreddits()
subreddit_list = "'AskReddit'"

db = pre.db_conn()
data = pre.db_query(db, subreddit_list, qry_lmt)
raw_corpus, corpus, labels, strata = pre.get_corpus(data)
print('corpus length: ' + str(len(corpus)))
print('corpus example: "' + str(corpus[1]) + '"')
print('labels length: ' + str(len(labels)))
print("labels: " + str(labels))

# ---------- Preparing training and test data ----------

max_features = 5000  # size of the vocabulary used
maxlen = 20  # length to which each sentence is padded

X = get_sequences(corpus, max_features, maxlen)

print('corpus example: "' + str(corpus[1]) + '"')
# print('sequence example: ' + str(seq[1]))
print('padded example: ' + str(X[1]))

y = get_labels_binary(labels, 1)

np.random.seed(1234)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# To test if crossvalidation works correctly, we can substitute random
# validation data
# y_test = np.random.binomial(1, 0.66, 200)

print("X_train :" + str(X_train))

print("X_train.shape: " + str(X_train.shape))
print("X_test.shape: " + str(X_test.shape))

print("y_train.shape: " + str(y_train.shape))
print("y_test.shape: " + str(y_test.shape))

print('y_train mean: ' + str(sum(y_train > 0)/float(y_train.shape[0])))
print('y_test mean: ' + str(sum(y_test > 0)/float(y_test.shape[0])))

# ---------- Logistic Regression benchmark ----------

# scikit learn logreg
# this will also return the predictions to make sure the model doesn't just
# predict one class only
print(lr_train(X_train, y_train, validation_data=(X_test, y_test)))

# keras simple logreg
print(lr_train(X_train, y_train, validation_data=(X_test, y_test), type='k1'))

# keras logreg with l1 and l2 regularization
print(lr_train(X_train, y_train, validation_data=(X_test, y_test), type='k2'))

# ---------- Convolutional Neural Network ----------

opt = SGD()
nb_filter = 100
batch_size = 32
filter_widths = [3, 5, 7]

# We will run the ConvNet with different filter sizes to determine the optimal
# filter width first
for filter_size in filter_widths:
    nn = cnn_build(max_features, maxlen, embedding_dim=100,
                   filter_size=filter_size, nb_filter=nb_filter,
                   dropout_p=0.25)
    cnn_train(nn, X_train, y_train, batch_size=batch_size, nb_epoch=1,
              validation_data=(X_test, y_test), val=True, opt=opt)

# After this we can tune other hyperparameters
# for example: nb_filter, dropout rate, embedding sizes, learning rate
# opt=SGD(lr=0.01)
# ....
