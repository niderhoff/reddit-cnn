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

The data are available at
[kaggle](https://www.kaggle.com/reddit/reddit-comments-may-2015).

Usage:

    $ python reddit_cnn.py

TODO:

*   data cleaning, this is _actually_ important
*   actually more than 1 subreddit?
*   init, activation
*   k-fold crossvalidation
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


def get_data(dataset="reddit", qry_lmt=25000, subreddit_list=pre.subreddits(),
             max_features=5000, maxlen=100, verbose=True):
    if (dataset is "reddit"):
        db = pre.db_conn()
        data = pre.db_query(db, subreddit_list, qry_lmt)
        raw_corpus, corpus, labels, strata = pre.get_corpus(data)
        print('corpus length: ' + str(len(corpus)))
        print('corpus example: "' + str(corpus[1]) + '"')
        print('labels length: ' + str(len(labels)))
        print("labels: " + str(labels))

        X = get_sequences(corpus, max_features, maxlen)

        print('corpus example: "' + str(corpus[1]) + '"')
        # print('sequence example: ' + str(seq[1]))
        print('padded example: ' + str(X[1]))

        y = get_labels_binary(labels, 1)

        np.random.seed(1234)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2)

        # To test if crossvalidation works correctly, we can substitute random
        # validation data
        #
        # y_test = np.random.binomial(1, 0.66, 200)
        if (verbose is True):
            print("X_train :" + str(X_train))
            print("X_train.shape: " + str(X_train.shape))
            print("X_test.shape: " + str(X_test.shape))
            print("y_train.shape: " + str(y_train.shape))
            print("y_test.shape: " + str(y_test.shape))
            print('y_train mean: ' + str(sum(y_train > 0) /
                                         float(y_train.shape[0])))
            print('y_test mean: ' + str(sum(y_test > 0) /
                                        float(y_test.shape[0])))

        return (X_train, X_test, y_train, y_test)
    elif (dataset is "imdb"):
        from keras.datasets import imdb
        (X_train, y_train), (X_test, y_test) = imdb.load_data(
                                                    nb_words=max_features)
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        if (verbose is True):
            print("X_train :" + str(X_train))
            print("X_train.shape: " + str(X_train.shape))
            print("X_test.shape: " + str(X_test.shape))
            print("y_train.shape: " + str(y_train.shape))
            print("y_test.shape: " + str(y_test.shape))
            print('y_train mean: ' + str(sum(y_train > 0) /
                                         float(y_train.shape[0])))
            print('y_test mean: ' + str(sum(y_test > 0) /
                                        float(y_test.shape[0])))
            print('X_train :' + str(X_train))
        return (X_train, X_test, y_train, y_test)


def cnn_build(max_features, maxlen, embedding_dim, filter_size, nb_filter,
              dropout_p, activation="relu", summary="full"):
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
    if (summary is "full"):
        print(nn.summary())
    elif (summary is "short"):
        summary = "MF " + str(max_features) + " | Len " + str(maxlen) + \
                  " | Embed " + str(embedding_dim) + " | F " + \
                  str(filter_size) + "x" + str(nb_filter) + " | Drop " + \
                  str(dropout_p) + " | " + activation
        print(summary)
    return nn


def cnn_train(model, X_train, y_train, validation_data=None, val=False,
              batch_size=32, nb_epoch=5, opt=SGD(), verbose=1):
    if (val is True and validation_data is not None):
        model.compile(loss='binary_crossentropy', optimizer=opt,
                      metrics=['accuracy'])
        if (verbose is 3):
            print("Optimizer set to " + str(opt))
            model.fit(X_train, y_train, batch_size=batch_size,
                      nb_epoch=nb_epoch, validation_data=validation_data,
                      verbose=0)
            print(model.evaluate(X_test, y_test, verbose=0))
        else:
            if (verbose is not 0):
                print("Optimizer set to " + str(opt))
            model.fit(X_train, y_train, batch_size=batch_size,
                      nb_epoch=nb_epoch, validation_data=validation_data,
                      verbose=verbose)
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
        if (val is True):
            print("Test fraction correct (LR-Accuracy) = {:.6f}".format(
                  lr.score(X_test, y_test)))
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
qry_lmt = 100000  # Actual number of posts we will be gathering.

# subreddits = subreddits()
subreddit_list = "'AskReddit'"

max_features = 5000  # size of the vocabulary used
maxlen = 20  # length to which each sentence is padded

X_train, X_test, y_train, y_test = get_data("reddit",
                                            qry_lmt=qry_lmt,
                                            subreddit_list=subreddit_list,
                                            max_features=max_features,
                                            maxlen=maxlen)

# or alternatively we can use IMDB dataset
X_train, X_test, y_train, y_test = get_data("imdb",
                                            max_features=max_features,
                                            maxlen=maxlen)

# ---------- Logistic Regression benchmark ----------

# Run a logistic regression benchmark first so we can later see if our
# ConvNet is somewhere in the ballpark.

# Scikit-learn logreg.
# This will also return the predictions to make sure the model doesn't just
# predict one class only
print(lr_train(X_train, y_train, validation_data=(X_test, y_test)))

# keras simple logreg
print(lr_train(X_train, y_train, validation_data=(X_test, y_test), type='k1'))

# keras logreg with l1 and l2 regularization
print(lr_train(X_train, y_train, validation_data=(X_test, y_test), type='k2'))

# ---------- Convolutional Neural Network ----------

# To find the optimal network structure, we will use the method proposed
# in http://arxiv.org/pdf/1510.03820v4.pdf (Zhang, Wallace 2016)

# First, start with a simple 1 layer CNN.

opt = SGD()
nb_filter = 100
batch_size = 32
filter_widths = range(1, 11)

# We will run the ConvNet with different filter sizes to determine the optimal
# filter width first
for filter_size in filter_widths:
    nn = cnn_build(max_features, maxlen, embedding_dim=100,
                   filter_size=filter_size, nb_filter=nb_filter,
                   dropout_p=0.25, summary="short")
    cnn_train(nn, X_train, y_train, batch_size=batch_size, nb_epoch=5,
              validation_data=(X_test, y_test), val=True, opt=opt, verbose=3)
    print("-------------------------------------------------------")

# After we decide for an appropriate filter size, we can try out using the
# same filter multiples times or several different size filters close to the
# optimal filter size.

filter_size = 3
print("Filter size permanently set to " + str(filter_size))

# However, I still have to find out how to code in multiple filter sizes in
# one model in keras.

# Now we can try out various feature map sizes, recommend is something between
# [100, 600]. If the optimal is close to the border, one should try going
# beyond that.
#
# for maxlen in range(100, 650, 50):
#    ...

# After this, we can try the effects of different activation functions.
# Recommended are for most data: Id(x), ReLU, or tanh. Id(x) mostly only
# works if you have one layer and we intent to increase that later so we should
# probably go with tanh or ReLU.
#
for activation in ('tanh', 'relu', 'linear'):
    nn = cnn_build(max_features, maxlen, embedding_dim=100,
                   filter_size=filter_size, nb_filter=nb_filter,
                   dropout_p=0.25, summary="short", activation=activation)
    cnn_train(nn, X_train, y_train, batch_size=batch_size, nb_epoch=5,
              validation_data=(X_test, y_test), val=True, opt=opt, verbose=3)
    print("-------------------------------------------------------")

activation = "relu"
print("Activation function permanently set to " + activation)

# 1-max-pooling is usually the best, but we can also try out k-max [5,10,15,20]
# ---> Stil have to figure out how to do k-max in keras.

# Also we can switch around with the dropout rate between (0.0, 0.5), but the
# impact this has will be very limited. Same goes for l2-regularization. We can
# add that into the model with a large constraint value. If we intent to
# increase the size of the feature map beond 600, we can add dropout > 0.5.
# However, all of this is pretty unnecessary if we only have 1 hidden layer.
#
for dropout_p in np.linspace(0, 0.5, 6):
    nn = cnn_build(max_features, maxlen, embedding_dim=100,
                   filter_size=filter_size, nb_filter=nb_filter,
                   dropout_p=dropout_p, summary="short", activation=activation)
    cnn_train(nn, X_train, y_train, batch_size=batch_size, nb_epoch=5,
              validation_data=(X_test, y_test), val=True, opt=opt, verbose=3)
    print("-------------------------------------------------------")

dropout_p = "0.25"
print("Dropout probability permanently set to " + dropout_p)

# After this we can fiddle around with the algorithm and learning rate. SGD
# works well with ReLU. Other options: Adadelta, Adam.
#
# opt=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# opt=Adadelta(lr=1.0)
# opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# ...

opt = SGD(lr=0.1)
print("Optimizer set to " + str(opt))
nn = cnn_build(max_features, maxlen, embedding_dim=100,
               filter_size=filter_size, nb_filter=nb_filter,
               dropout_p=dropout_p, summary="short", activation=activation)
cnn_train(nn, X_train, y_train, batch_size=batch_size, nb_epoch=5,
          validation_data=(X_test, y_test), val=True, opt=opt, verbose=1)
print("-------------------------------------------------------")
