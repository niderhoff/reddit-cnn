#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reddit CNN - binary classification on reddit comment scores.

Reads posts and scores from reddit comment database, provide sequence
embeddings for the comment text to feed into various machine learning models.

Provides functions to train logistic regression from scikit-learn (simple) as
well as keras (simple and with l1 and l2 regularization).

Also trains Convolutional Neural Networks (CNN) with varying filter sizes,
filter numbers and optimizers using keras (theano).

Can be supplied with a range of Filter sizes, Dropout Rates, and activation
functions. The program will then calculate either all possible model
combinations (if --perm is supplied as an argument) or simple models each for
the different filter sizes, dropout rates, or activation functions. The --perm
option can be used in conjunction with --parallel to use all supplied filter
sizes in a single model instead of one per model. This is useful as a second
step after the roughly optimal filter size has been identified and the user
wants to add several filter sizes close to the optimum.

Usage:

    $ reddit_cnn.py [-h] [--dataset DATASET] [-q QRY_LMT]
                     [--max_features MAX_FEATURES] [--maxlen MAXLEN]
                     [--minlen MINLEN] [-b BATCH_SIZE] [-o OPT] [-e EPOCHS]
                     [-N NB_FILTER] [-F [FILTERS [FILTERS ...]]]
                     [-A [ACTIVATION [ACTIVATION ...]]]
                     [-D [DROPOUT [DROPOUT ...]]] [-E EMBED] [--perm]
                     [--logreg] [--dry] [--parallel] [--cm] [-v VERBOSE]
                     [-f FILE]

Args:

    -h, --help            show this help message and exit
    --dataset DATASET     dataset to be used (default: 'reddit')
    -q QRY_LMT, --qry_lmt QRY_LMT
                        amount of data to query (default: 10000)
    --max_features MAX_FEATURES
                        size of vocabulary (default: 5000)
    --maxlen MAXLEN       maximum comment length (default: 100)
    --minlen MINLEN       minimum comment length (default: 0)
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size (default: 32)
    -o OPT, --opt OPT     optimizer flag (default: 'rmsprop')
    -e EPOCHS, --epochs EPOCHS
                        number of epochs for models (default: 5)
    -N NB_FILTER, --nb_filter NB_FILTER
                        number of filters for each size (default: 100)
    -F [FILTERS [FILTERS ...]], --filters [FILTERS [FILTERS ...]]
                        filter sizes to be calculated (default: 3)
    -A [ACTIVATION [ACTIVATION ...]], --activation [ACTIVATION [...]]
                        activation functions to use (default: ['relu'])
    -D [DROPOUT [DROPOUT ...]], --dropout [DROPOUT [DROPOUT ...]]
                        dropout percentages (default: [0.25])
    -E EMBED, --embed EMBED
                        embedding dimension (default: 100)
    --perm                calculate all possible model Permutations (default:
                        True)
    --logreg              calculate logreg benchmark? (default: False)
    --dry                 do not actually calculate anything (default: False)
    --parallel            run filter sizes in parallel (default: False)
    --cm                  calculates confusion matrix (default: False)
    -v VERBOSE, --verbose VERBOSE
                        verbosity between 0 and 3 (default: 2)
    -f FILE, --file FILE  file to output to (default: None)

The data are available at
[kaggle](https://www.kaggle.com/reddit/reddit-comments-may-2015).

TODO:

*   actually more than 1 subreddit?
*   init, activation
*   k-fold crossvalidation
*   add docstrings to everything
*   add option to display+print model plot
*   add option to time benchmark
*   catch argument exceptions
*   update documentation
"""

from __future__ import print_function
import argparse
import sys
from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.regularizers import l1, l2, l1l2
# from keras.utils.visualize_util import plot
from keras.datasets import imdb

import preprocess as pre


# ---------- General purpose functions ----------
def query_yes_no(question, default="yes"):
    """
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    From http://code.activestate.com/recipes/577058/.
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


# ---------- Data prep ----------
def get_sequences(corpus, max_features=5000, seqlen=100):
    """
    Will convert the corpus to a word index and pad the resulting
    sequences to the maximum length if they fall short.

    Args:
        corpus: the corpus of comments
        max_features: number of words in the word index. words that are used
                      less frequently will be replaced by zeroes. (?)
        maxlen: maximum length of comments. longer ones will be truncated,
                shorter ones will be padded with zeroes on the left hand side.

    Returns:
        data matrix X
    """
    tokenized = Tokenizer(nb_words=max_features)
    tokenized.fit_on_texts(corpus)
    seq = tokenized.texts_to_sequences(corpus)
    X = sequence.pad_sequences(seq, seqlen)
    return X


def get_labels_binary(labels, threshold=1):
    """
    Will turn the labels (reddit comment karma) into binary classes depending
    on a given threshold.

    Args:
        labels: the list of karma scores
        threshold: value at wich to split the scores into classes

    Returns:
        np.array with binary classes
    """
    Y = np.asarray(labels)
    Ybool = abs(Y) > threshold
    Ybin = Ybool.astype(int)
    if (verbose > 0):
        print('Y.shape: ' + str(Y.shape))
        if (verbose == 3):
            print('Y: ' + str(Y))
            print("Y > 1: " + str(Ybool))
            print("Y binary: " + str(Ybin))

    return np.expand_dims(Ybin, axis=1)


def get_data(dataset="reddit", qry_lmt=25000, subreddit_list=pre.subreddits(),
             max_features=5000, minlen=5, maxlen=100, seqlen=100,
             minscore=None, maxscore=None, split=0.2, verbose=1):
    if (dataset.lower() == "reddit"):
        raw_corpus, corpus, labels, strata = pre.get_corpora(
            subreddit_list, qry_lmt, minlen=minlen, maxlen=maxlen,
            minscore=minscore, maxscore=maxscore,
            batch_size=qry_lmt/10, verbose=verbose)
        X = get_sequences(corpus, max_features, seqlen)
        y = get_labels_binary(labels, 1)

        np.random.seed(1234)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=split)

        # To test if crossvalidation works correctly, we can substitute random
        # validation data
        #
        # y_test = np.random.binomial(1, 0.66, 200)
        if (verbose > 1):
            print('Using reddit dataset.')
            print('corpus length: ' + str(len(corpus)))
            print('corpus example: "' + str(corpus[1]) + '"')
            print('labels length: ' + str(len(labels)))
            print('corpus example: "' + str(corpus[1]) + '"')
            print('padded example: ' + str(X[1]))
            if (verbose > 2):
                print("labels: " + str(labels))
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
    elif (dataset.lower() == "imdb"):
        (X_train, y_train), (X_test, y_test) = imdb.load_data(
                                                    nb_words=max_features)
        X_train = sequence.pad_sequences(X_train, maxlen=seqlen)
        X_test = sequence.pad_sequences(X_test, maxlen=seqlen)
        if (verbose > 1):
            if (verbose > 2):
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
    else:
        print(dataset + " is not a valid dataset.")


# ---------- All the models we can call ----------
def cnn_simple(max_features, seqlen, embedding_dim, filter_size, nb_filter,
               dropout_p, activation="relu", summary="full", l2reg=None,
               l1reg=None, batchnorm=None, layers=1):
    nn = Sequential()
    nn.add(Embedding(input_dim=max_features, output_dim=embedding_dim,
                     input_length=seqlen))
    nn.add(Dropout(dropout_p))
    nn.add(Convolution1D(
        nb_filter,
        filter_size,
        activation=activation
        ))
    nn.add(MaxPooling1D(pool_length=seqlen - filter_size + 1))
    nn.add(Flatten())
    nn.add(Dropout(dropout_p))
    if (l1reg is not None and l1reg is float and l2reg is not None and l2reg is
            float):
        nn.add(Dense(1), W_regularizer=l1l2(l1reg, l2reg))
    elif (l2reg is not None and l2reg is float):
        nn.add(Dense(1), W_regularizer=l2(l2reg))
    elif (l1reg is not None and l1reg is float):
        nn.add(Dense(1), W_regularizer=l1(l1reg))
    else:
        nn.add(Dense(1))
    if (batchnorm is True):
        nn.add(BatchNormalization())
    nn.add(Activation('sigmoid'))
    if (summary == "full"):
        print(nn.summary())
    elif (summary == "short"):
        summary = "MF " + str(max_features) + " | Len " + str(seqlen) + \
                  " | Embed " + str(embedding_dim) + " | F " + \
                  str(filter_size) + "x" + str(nb_filter) + " | Drop " + \
                  str(dropout_p) + " | " + activation
        print(summary)
    return nn


def cnn_train_simple(model, X_train, y_train, validation_data=None, val=False,
                     batch_size=32, nb_epoch=5, opt=SGD(), verbose=1):
    if (val is True and validation_data is not None):
        X_test, y_test = validation_data
        model.compile(loss='binary_crossentropy', optimizer=opt,
                      metrics=['accuracy'])
        if (verbose is 1):
            model.fit(X_train, y_train, batch_size=batch_size,
                      nb_epoch=nb_epoch, validation_data=validation_data,
                      verbose=0)
            print(model.evaluate(X_test, y_test, verbose=0))
        elif (verbose is 2):
            model.fit(X_train, y_train, batch_size=batch_size,
                      nb_epoch=nb_epoch, validation_data=validation_data,
                      verbose=2)
        elif (verbose is 3):
            model.fit(X_train, y_train, batch_size=batch_size,
                      nb_epoch=nb_epoch, validation_data=validation_data,
                      verbose=1)
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt)
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
        return(model)


def cnn_parallel(max_features, seqlen, embedding_dim, ngram_filters, nb_filter,
                 dropout_p, activation="relu", summary="full", l2reg=None,
                 l1reg=None, batchnorm=None):
    conv_filters = []
    for n_gram in ngram_filters:
        sequential = Sequential()
        conv_filters.append(sequential)
        sequential.add(Embedding(input_dim=max_features,
                                 output_dim=embedding_dim,
                                 input_length=seqlen))
        sequential.add(Dropout(dropout_p))
        sequential.add(Convolution1D(nb_filter, n_gram, activation=activation))
        sequential.add(MaxPooling1D(pool_length=seqlen - n_gram + 1))
        sequential.add(Flatten())
    nn = Sequential()
    nn.add(Merge(conv_filters, mode='concat'))
    nn.add(Dropout(dropout_p))
    if (l1reg is not None and l1reg is float and l2reg is not None and l2reg is
            float):
        nn.add(Dense(1), W_regularizer=l1l2(l1reg, l2reg))
    elif (l2reg is not None and l2reg is float):
        nn.add(Dense(1), W_regularizer=l2(l2reg))
    elif (l1reg is not None and l1reg is float):
        nn.add(Dense(1), W_regularizer=l1(l1reg))
    else:
        nn.add(Dense(1))
    if (batchnorm is True):
        nn.add(BatchNormalization())
    nn.add(Activation("sigmoid"))
    if (summary == "full"):
        print(nn.summary())
    elif (summary == "short"):
        summary = "MF " + str(max_features) + " | Len " + str(seqlen) + \
                  " | Embed " + str(embedding_dim) + " | F " + \
                  str(ngram_filters) + "x" + str(nb_filter) + " | Drop " + \
                  str(dropout_p) + " | " + activation
        print(summary)
    return nn


def cnn_train_parallel(model, X_train, y_train, ngram_filters,
                       validation_data=None, val=False,
                       batch_size=32, nb_epoch=5, opt=SGD(), verbose=1):
    X_test, y_test = validation_data
    concat_X_test = []
    concat_X_train = []
    for i in range(len(ngram_filters)):
        concat_X_test.append(X_test)
        concat_X_train.append(X_train)

    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    if (verbose is 1):
        model.fit(concat_X_train, y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(concat_X_test, y_test), verbose=0)
        print(model.evaluate(concat_X_test, y_test, verbose=0))
    elif (verbose is 2):
        model.fit(concat_X_train, y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(concat_X_test, y_test), verbose=2)
    elif (verbose is 3):
        model.fit(concat_X_train, y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(concat_X_test, y_test), verbose=1)
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt)
        model.fit(concat_X_train, y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch)
        return(model)


# ---------- Diagnostics and Benchmarks ----------
def lr_train(X_train, y_train, val=True, validation_data=None, type='skl',
             nb_epoch=10, reg=l1l2(l1=0.01, l2=0.01), verbose=1):
    X_test, y_test = validation_data
    if (type == 'skl'):
        lr = LogisticRegressionCV()
        lr.fit(X_train, y_train.ravel())
        pred_y = lr.predict(X_test)
        if (val is True and verbose > 0):
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


def print_cm(nn, X_test, y_test, batch_size=32):
    print("Confusion Matrix (frequency, normalized):")
    y_pred = nn.predict_classes(X_test, batch_size=batch_size, verbose=0)
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    np.set_printoptions(precision=3, suppress=True)
    print(cm)
    print(cm_normalized)


# ---------- Parsing command line arguments ----------
parser = argparse.ArgumentParser(
    description='Reddit CNN - binary classification on reddit comment scores.')

# 'imdb' or 'reddit' dataset?
parser.add_argument('--dataset', default='reddit',
                    help='dataset to be used (default: \'reddit\')')

# Post fetching
parser.add_argument('-q', '--qry_lmt', default=10000, type=int,
                    help='amount of data to query (default: 10000)')
parser.add_argument('--max_features', default=5000, type=int,
                    help='size of vocabulary (default: 5000)')
parser.add_argument('--seqlen', default=100, type=int,
                    help='length to which sequences will be padded \
                    (default: 100)')
parser.add_argument('--maxlen', default=100, type=int,
                    help='maximum comment length (default: 100)')
parser.add_argument('--minlen', default=0, type=int,
                    help='minimum comment length (default: 0)')

# General Hyperparameters
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='batch size (default: 32)')

# --opt will expect a keras.optimizer call
parser.add_argument('-o', '--opt', default='rmsprop',
                    help='optimizer flag (default: \'rmsprop\')')
parser.add_argument('-e', '--epochs', default=5, type=int,
                    help='number of epochs for models (default: 5)')

# Model Parameters
parser.add_argument('-N', '--nb_filter', default=100, type=int,
                    help='number of filters for each size (default: 100)')
# --filters will expect a list of integers corresponding to selected filter
# sizes, e.g. -F 3 5 7.
parser.add_argument('-F', '--filters', nargs='*', default=[3], type=int,
                    help='filter sizes to be calculated (default: 3)')
# --activation will expect strings corresponding to keras activation
# functions, e.g. -A 'tanh' 'relu'.
parser.add_argument('-A', '--activation', nargs='*', default=['relu'],
                    help='activation functions to use (default: [\'relu\'])')

# Regularization
# --dropout will expect a list of floats corresponding to different dropout
# rates, e.g. -D 0.1 0.25 0.5
parser.add_argument('-D', '--dropout', nargs='*', default=[0.25], type=float,
                    help='dropout percentages (default: [0.25])')
parser.add_argument('-E', '--embed', default=100, type=int,
                    help='embedding dimension (default: 100)')
parser.add_argument('-l1', default=None, type=float,
                    help='l1 regularization for penultimate layer')
parser.add_argument('-l2', default=None, type=float,
                    help='l2 regularization for penultimate layer')
parser.add_argument('-s', '--split', default=0.2, type=float,
                    help='train/test split ratio (default: 0.1)')
# Batchnorm is not working with cuda 4
parser.add_argument('--batchnorm', default=False, action='store_true',
                    help='add Batch Normalization to activations')

# Multiple layers not working as of yet.
# parser.add_argument('-L', '--layers', default=1, type=int,
#                     help='number of convolutional layers (default=1)')
parser.add_argument('--model', default="simple",
                    help="the type of model (simple/parallel/twolayer)")

# Switches
# For now this switch is always true.
parser.add_argument('--perm', default=True, action='store_true',
                    help='calculate all possible model Permutations \
                    (default: True)')
parser.add_argument('--logreg', action='store_true',
                    help='calculate logreg benchmark? (default: False)')
parser.add_argument('--dry', default=False, action='store_true',
                    help='do not actually calculate anything (default: False)')
parser.add_argument('--cm', default=False, action='store_true',
                    help='calculates confusion matrix (default: False)')

# Other arguments
parser.add_argument('-v', '--verbose', default=2, type=int,
                    help='verbosity between 0 and 3 (default: 2)')
# parser.add_argument('-f', '--file', default=None,
#                     help='file to output to (default: None)')

args = parser.parse_args()

# ---------- Store command line argument variables ----------
# Verbosity levels
# 0: nothing
# 1: only endresults per model
# 2: short summaries and results per epoch
# 3: debug infos (data matrices, examples of data), live progress bar reports
#    per epoch
verbose = args.verbose
if (verbose > 0):
    print("verbosity level: " + str(verbose))

# Dry run yes/no?
dry_run = args.dry

# Calculates all possible permutations from the model options selected
perm = args.perm

# Calculate confusion matrix?
cm = args.cm

# Train/Test split size
split = args.split

# Model selection
model = args.model

# Hyperparameter constants
nb_filter = args.nb_filter
batch_size = args.batch_size
opt = args.opt
nb_epoch = args.epochs
embedding_dim = args.embed
l1reg = args.l1
l2reg = args.l2
batchnorm = args.batchnorm

# Hyperparameter lists
filter_widths = args.filters
activation_list = args.activation
dropout_list = args.dropout

# Permanent hyperparameters
dropout_p = dropout_list[0]
activation = activation_list[0]
filter_size = filter_widths[0]

# TODO: check arguments for exceptions

# ---------- Data gathering ----------
qry_lmt = args.qry_lmt  # Actual number of posts we will be gathering.

# subreddits = subreddits()
subreddit_list = "'AskReddit'"

max_features = args.max_features  # size of the vocabulary used
seqlen = args.seqlen  # length to which each sentence is padded
maxlen = args.maxlen  # maximum length of comment to be considered
minlen = args.minlen  # minimum length of a comment to be considered

if (seqlen > maxlen):
    print("padding length is greater than actual length of the comments.")
    print("setting padding length (" + str(seqlen) + ") to " + str(maxlen))

X_train, X_test, y_train, y_test = get_data(args.dataset,
                                            qry_lmt=qry_lmt,
                                            subreddit_list=subreddit_list,
                                            max_features=max_features,
                                            maxlen=maxlen, minlen=minlen,
                                            seqlen=seqlen,
                                            verbose=verbose, split=split)

print("======================================================")

# ---------- Logistic Regression benchmark ----------

# Run a logistic regression benchmark first so we can later see if our
# ConvNet is somewhere in the ballpark.

if (args.logreg is True):
    if (verbose > 0):
        print("Running logistic regression benchmarks.")
    if (dry_run is False):
        # Scikit-learn logreg.
        # This will also return the predictions to make sure the model doesn't
        # just predict one class only
        print(lr_train(X_train, y_train, validation_data=(X_test, y_test)))
        print("------------------------------------------------------")

        # keras simple logreg
        print(lr_train(X_train, y_train, validation_data=(X_test, y_test),
              type='k1'))
        print("------------------------------------------------------")

        # keras logreg with l1 and l2 regularization
        print(lr_train(X_train, y_train, validation_data=(X_test, y_test),
              type='k2'))
        print("------------------------------------------------------")
    print("======================================================")

# ---------- Convolutional Neural Network ----------

# To find the optimal network structure, we will use the method proposed
# in http://arxiv.org/pdf/1510.03820v4.pdf (Zhang, Wallace 2016)

# First, start with a simple 1 layer CNN.
if (verbose > 0):
    print("Optimizer permanently set to " + str(opt))
    print("Embedding dim permanently set to " + str(embedding_dim))
    print("Number of filters set to " + str(nb_filter))
    print("Batch size set to " + str(batch_size))
    print("Number of epochs set to " + str(nb_epoch))
    if (verbose == 3):
        summary = "full"
    else:
        summary = "short"
else:
    summary = "none"


if (perm is True):
    if (verbose > 0):
        print("Selected filter sizes " + str(filter_widths))
        print("Selected Dropout rates " + str(dropout_list))
        print("Selected Activation functions " + str(activation_list))
        print("Calculating all possible model permutations")
    if (model == "parallel" and len(filter_widths) > 1):
        if (verbose > 0):
            print("Selected parallel filters.")
        s = [dropout_list, activation_list]
        models = list(product(*s))
        M = len(models)
        print("Found " + str(M) + " possible models.")
        if (query_yes_no("Do you wish to continue?")):
            for m in models:
                nn = cnn_parallel(max_features=max_features, seqlen=seqlen,
                                  embedding_dim=embedding_dim,
                                  ngram_filters=filter_widths,
                                  nb_filter=nb_filter,
                                  dropout_p=m[0],
                                  activation=m[1],
                                  l2reg=l2reg,
                                  l1reg=l1reg,
                                  batchnorm=batchnorm,
                                  summary=summary)
                if (dry_run is False):
                    cnn_train_parallel(nn, X_train, y_train,
                                       ngram_filters=filter_widths,
                                       validation_data=(X_test, y_test),
                                       val=True,
                                       batch_size=batch_size,
                                       nb_epoch=nb_epoch,
                                       opt=opt,
                                       verbose=verbose)
                    if (cm is True):
                        # Confusion Matrix code
                        concat_X_test = []
                        for i in range(len(filter_widths)):
                            concat_X_test.append(X_test)
                        print_cm(nn, concat_X_test, y_test)
                print("------------------------------------------------------")
        else:
            print("Aborting.")
            print("======================================================")
    elif (model == "simple"):
        s = [filter_widths, dropout_list, activation_list]
        models = list(product(*s))
        M = len(models)
        print("Found " + str(M) + " possible models.")

        if (query_yes_no("Do you wish to continue?")):
            for m in models:
                nn = cnn_simple(max_features=max_features, seqlen=seqlen,
                                embedding_dim=embedding_dim,
                                filter_size=m[0],
                                nb_filter=nb_filter,
                                dropout_p=m[1],
                                activation=m[2],
                                l2reg=l2reg,
                                l1reg=l1reg,
                                batchnorm=batchnorm,
                                summary=summary)
                if (dry_run is False):
                    cnn_train_simple(nn, X_train, y_train,
                                     batch_size=batch_size,
                                     nb_epoch=nb_epoch,
                                     validation_data=(X_test, y_test),
                                     val=True,
                                     opt=opt,
                                     verbose=verbose)
                    if (cm is True):
                        # Confusion Matrix code
                        print_cm(nn)
                print("------------------------------------------------------")
    elif (model == "twolayer"):
        s = [filter_widths, dropout_list, activation_list]
        models = list(product(*s))
        M = len(models)
        print("Found " + str(M) + " possible models.")

        if (query_yes_no("Do you wish to continue?")):
            for m in models:
                nn = cnn_twolayer(max_features=max_features, seqlen=seqlen,
                                  embedding_dim=embedding_dim,
                                  filter_size=m[0],
                                  nb_filter=nb_filter,
                                  dropout_p=m[1],
                                  activation=m[2],
                                  l2reg=l2reg,
                                  l1reg=l1reg,
                                  batchnorm=batchnorm,
                                  summary=summary)
                if (dry_run is False):
                    cnn_train_twolayer(nn, X_train, y_train,
                                       batch_size=batch_size,
                                       nb_epoch=nb_epoch,
                                       validation_data=(X_test, y_test),
                                       val=True,
                                       opt=opt,
                                       verbose=verbose)
                    if (cm is True):
                        # Confusion Matrix code
                        print_cm(nn)
                print("------------------------------------------------------")
    else:
        print("The model selected is not known to the program: " +
              str(model) + ". Aborting.")
        print("Or didn't give sufficient parameters for the model. (Filters?)")
        print("======================================================")
else:
    if (verbose > 0):
        print('You chose not to calculate all model permutations. ' +
              'Running default routine.')
        print("Selected filter sizes " + str(filter_widths))
        print("Selected Dropout rates " + str(dropout_list))
        print("Selected Activation functions " + str(activation_list))

# We will run the ConvNet with different filter sizes to determine the
# optimal filter width first
    for fs in filter_widths:
        nn = cnn_simple(max_features, seqlen, embedding_dim=embedding_dim,
                        filter_size=fs, nb_filter=nb_filter,
                        dropout_p=dropout_p, summary=summary)
        if (dry_run is False):
            cnn_train_simple(nn, X_train, y_train, batch_size=batch_size,
                             nb_epoch=nb_epoch,
                             validation_data=(X_test, y_test), val=True,
                             opt=opt, verbose=verbose)
        print("------------------------------------------------------")

# After this, we can try the effects of different activation functions.
# Recommended are for most data: Id(x), ReLU, or tanh. Id(x) mostly only
# works if you have one layer and we intent to increase that later so we should
# probably go with tanh or ReLU.
#
    for act in activation_list:
        nn = cnn_simple(max_features, seqlen, embedding_dim=embedding_dim,
                        filter_size=filter_size, nb_filter=nb_filter,
                        dropout_p=dropout_p, summary=summary, activation=act)
        if (dry_run is False):
            cnn_train_simple(nn, X_train, y_train, batch_size=batch_size,
                             nb_epoch=nb_epoch,
                             validation_data=(X_test, y_test), val=True,
                             opt=opt, verbose=verbose)
        print("------------------------------------------------------")

# Also we can switch around with the dropout rate between (0.0, 0.5), but the
# impact this has will be very limited. Same goes for l2-regularization. We can
# add that into the model with a large constraint value. If we intent to
# increase the size of the feature map beond 600, we can add dropout > 0.5.
# However, all of this is pretty unnecessary if we only have 1 hidden layer.
#
    for p in dropout_list:
        nn = cnn_simple(max_features, seqlen, embedding_dim=embedding_dim,
                        filter_size=filter_size, nb_filter=nb_filter,
                        dropout_p=p, summary=summary, activation=activation)
        if (dry_run is False):
            cnn_train_simple(nn, X_train, y_train, batch_size=batch_size,
                             nb_epoch=nb_epoch,
                             validation_data=(X_test, y_test), val=True,
                             opt=opt, verbose=verbose)
        print("------------------------------------------------------")

# After we decide for an appropriate filter size, we can try out using the
# same filter multiples times or several different size filters close to the
# optimal filter size.

# filter_size = 3
# print("Filter size permanently set to " + str(filter_size))

# However, I still have to find out how to code in multiple filter sizes in
# one model in keras.

# Now we can try out various feature map sizes, recommend is something between
# [100, 600]. If the optimal is close to the border, one should try going
# beyond that.
#
# for seqlen in range(100, 650, 50):
#    ...

# 1-max-pooling is usually the best, but we can also try out k-max [5,10,15,20]
# ---> Stil have to figure out how to do k-max in keras.

# After this we can fiddle around with the algorithm and learning rate. SGD
# works well with ReLU. Other options: Adadelta, Adam.
#
# opt=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# opt=Adadelta(lr=1.0)
# opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# ...

# opt = SGD(lr=0.1)
# print("Optimizer set to " + str(opt))
# nn = cnn_build(max_features, seqlen, embedding_dim=100,
#                filter_size=filter_size, nb_filter=nb_filter,
#                dropout_p=dropout_p, summary="short", activation=activation)
# cnn_train(nn, X_train, y_train, batch_size=batch_size, nb_epoch=5,
#           validation_data=(X_test, y_test), val=True, opt=opt, verbose=1)
# print("------------------------------------------------------")

    print("======================================================")
