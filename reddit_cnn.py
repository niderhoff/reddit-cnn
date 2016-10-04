#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reddit CNN - binary classification on reddit comment scores.

Reads posts and scores from reddit comment database, provide sequence
embeddings for the comment text to feed into various machine learning models.

The comments can be filtered by subreddit, karma score, length and an option
can be selected to always create a balanced dataset (i.e. pos. and neg. karma
is evenly distributed).

Provides functions to train logistic regression from scikit-learn (simple) as
well as keras (simple and with l1 and l2 regularization).

Also trains Convolutional Neural Networks (CNN) with varying filter sizes,
filter numbers using keras (theano). Options include different
optimizers, l1 and l2 regularization, batch normalization. Will try to run on
GPU if possible.

Can be supplied with a range of Filter sizes, Dropout Rates, and activation
functions. The program will then calculate either all possible model
combinations (if --perm is supplied as an argument) or simple models each for
the different filter sizes, dropout rates, or activation functions. The --perm
option can be used in conjunction with --parallel to use all supplied filter
sizes in a single model instead of one per model. This is useful as a second
step after the roughly optimal filter size has been identified and the user
wants to add several filter sizes close to the optimum.

Usage:

    $ python reddit_cnn.py [-h] [--dataset DATASET] [-q QRY_LMT]
                     [--max_features MAX_FEATURES] [--seqlen SEQLEN]
                     [--maxlen MAXLEN] [--minlen MINLEN]
                     [--scorerange SCORERANGE SCORERANGE] [--negrange]
                     [--balanced] [-b BATCH_SIZE] [-o OPT] [-e EPOCHS]
                     [-N NB_FILTER] [-F [FILTERS [FILTERS ...]]]
                     [-A [ACTIVATION [ACTIVATION ...]]]
                     [-D [DROPOUT [DROPOUT ...]]] [-E EMBED] [-l1 L1] [-l2 L2]
                     [-s SPLIT] [--batchnorm] [--model MODEL] [--perm]
                     [--logreg] [--dry] [--cm] [-v VERBOSE]
                     [--fromfile FROMFILE]

Args:

    -h, --help            show help message and exit
    --dataset DATASET     dataset to be used (default: 'reddit')
    -q QRY_LMT, --qry_lmt QRY_LMT
                          amount of data to query (default: 10000)
    --max_features MAX_FEATURES
                          size of vocabulary (default: 5000)
    --seqlen SEQLEN       length to which sequences will be padded (default:
                          100)
    --maxlen MAXLEN       maximum comment length (default: 100)
    --minlen MINLEN       minimum comment length (default: 0)
    --scorerange SCORERANGE SCORERANGE
    --negrange
    --balanced
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                          batch size (default: 32)
    -o OPT, --opt OPT     optimizer flag (default: 'rmsprop')
    -e EPOCHS, --epochs EPOCHS
                          number of epochs for models (default: 5)
    -N NB_FILTER, --nb_filter NB_FILTER
                          number of filters for each size (default: 100)
    -F [FILTERS [FILTERS ...]], --filters [FILTERS [FILTERS ...]]
                          filter sizes to be calculated (default: 3)
    -A [ACTIVATION [ACTIVATION ...]], --activation [ACTIVATION ACTIVATION ...]
                          activation functions to use (default: ['relu'])
    -D [DROPOUT [DROPOUT ...]], --dropout [DROPOUT [DROPOUT ...]]
                          dropout percentages (default: [0.25])
    -E EMBED, --embed EMBED
                          embedding dimension (default: 100)
    -l1 L1                l1 regularization for penultimate layer
    -l2 L2                l2 regularization for penultimate layer
    -s SPLIT, --split SPLIT
                          train/test split ratio (default: 0.1)
    --batchnorm           add Batch Normalization to activations
    --model MODEL         the type of model (simple/parallel/twolayer)
    --perm                calculate all possible model Permutations (default:
                          True)
    --logreg              calculate logreg benchmark? (default: False)
    --dry                 do not actually calculate anything (default: False)
    --cm                  calculates confusion matrix (default: False)
    -v VERBOSE, --verbose VERBOSE
                          verbosity between 0 and 3 (default: 2)
    --fromfile FROMFILE   file to read datamatrix from (default: None)
                          (currently not in use)

The data are available at
[kaggle](https://www.kaggle.com/reddit/reddit-comments-may-2015).

TODO:

*   add option so you draw the plots to file instead of displayed them
*   also add option to have files named after models if you calculate multiple
*   add option to cut the file-input using qry_lmt
*   possibility to randomize selection from subreddits (as of now it will
    fill all data from first subreddit found if possible)
*   implement non-random initialization for model
*   implement k-fold crossvalidation
*   add docstrings to all functions
*   update existing docstrings
*   add option to time benchmark
*   outsource code for logreg, and time benchmarks
*   catch commandline argument exceptions properly

Known Bugs and Limitations:

*   BATCH NORMALIZATION not working on CUDA v4. (this is an external issue that
    can not be fixed. however, one could think of implementing a check for the
    CUDA version.)
*   SGD optimizer cannot be called through command line (since keras expects
    an actual SGD() call and not a string
*   Newest iteration of database code is really slow.
*   Some documentation is false or misleading.
*   The current implementation of the CNN-model might not be the best.
"""

from __future__ import print_function
import argparse
import sys
import os.path
from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense
from keras.regularizers import l1l2
from keras.datasets import imdb

import preprocess as pre
import vis
from models.cnn import CNN_Simple, CNN_TwoLayer, CNN_TwoLayer2, CNN_Parallel
from models.cnn import CNN_ThreeLayer


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
    Ybool = Y > threshold
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
             scorerange=None, negrange=False, split=0.2, verbose=1,
             balanced=False, fromfile=None):
    if (fromfile is not None and os.path.isfile(fromfile)):
        f = np.load(fromfile)
        raw_corpus, corpus, labels, strata = (f['raw_corpus'], f['corpus'],
                                              f['labels'], f['strata'])
        X = get_sequences(corpus, max_features, seqlen)
        y = get_labels_binary(labels, 1)

        if (args.noseed is not True):
            np.random.seed(1234)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=split)
        if (verbose > 1):
            print('Using dataset from file: ' + str(fromfile))
            print_data_info(corpus, labels, X_train, X_test, y_train, y_test)
            print('padded example: ' + str(X[1]))
        return (X_train, X_test, y_train, y_test)
    elif (dataset.lower() == "reddit"):
        raw_corpus, corpus, labels, strata = pre.build_corpus(
            subreddit_list, qry_lmt, minlen=minlen, maxlen=maxlen,
            scorerange=scorerange, negrange=negrange,
            batch_size=qry_lmt/10, verbose=verbose, balanced=balanced)
        X = get_sequences(corpus, max_features, seqlen)
        y = get_labels_binary(labels, 1)

        if (args.noseed is not True):
            np.random.seed(1234)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=split)

        # To test if crossvalidation works correctly, we can substitute random
        # validation data
        #
        # y_test = np.random.binomial(1, 0.66, 200)
        if (verbose > 1):
            print('Using reddit dataset.')
            print_data_info(corpus, labels, X_train, X_test, y_train, y_test)
            print('padded example: ' + str(X[1]))
        return (X_train, X_test, y_train, y_test)
    elif (dataset.lower() == "imdb"):
        (X_train, y_train), (X_test, y_test) = imdb.load_data(
                                                    nb_words=max_features)
        X_train = sequence.pad_sequences(X_train, maxlen=seqlen)
        X_test = sequence.pad_sequences(X_test, maxlen=seqlen)
        if (verbose > 1):
            print('Using imdb dataset.')
            if (verbose > 2):
                print("X_train :" + str(X_train))
            print_data_info(corpus, labels, X_train, X_test, y_train, y_test)
        return (X_train, X_test, y_train, y_test)
    else:
        print(dataset + " is not a valid dataset.")


def print_data_info(corpus, labels, X_train, X_test, y_train, y_test):
    print('corpus length: ' + str(len(corpus)))
    print('corpus example: "' + str(corpus[1]) + '"')
    print('labels length: ' + str(len(labels)))
    print('corpus example: "' + str(corpus[1]) + '"')
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
        print("min score: " + str(min(labels)))
        print("max score: " + str(max(labels)))
        print("min length: " + str(min(map(len, corpus))))
        print("max length: " + str(max([len(x.split()) for x in corpus])))


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
    print(y_pred)
    print(sum(y_pred)/len(y_pred))
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
parser.add_argument('--subreddits', default=None, nargs='*',
                    help='list of subreddits (default: None)')
parser.add_argument('--noseed', default=False, action='store_true')

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
parser.add_argument('--scorerange', nargs=2, default=None, type=int)
parser.add_argument('--negrange', default=False, action='store_true')
parser.add_argument('--balanced', default=False, action='store_true')

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
# parser.add_argument('--perm', default=True, action='store_true',
#                     help='calculate all possible model Permutations \
#                     (default: True)')
parser.add_argument('--logreg', action='store_true',
                    help='calculate logreg benchmark? (default: False)')
parser.add_argument('--dry', default=False, action='store_true',
                    help='do not actually calculate anything (default: False)')
parser.add_argument('--cm', default=False, action='store_true',
                    help='calculates confusion matrix (default: False)')
parser.add_argument('--plot', default=False, action='store_true',
                    help='plot the model metrics (default: False)')

# Other arguments
parser.add_argument('-v', '--verbose', default=2, type=int,
                    help='verbosity between 0 and 3 (default: 2)')
parser.add_argument('-f', '--outfile', default=None,
                    help='file to output results to (default: None)')
parser.add_argument('--fromfile', default=None,
                    help="file input (default: None)")

args = parser.parse_args()


# ---------- Logging ----------
# Initialize logfile if required
if (args.outfile is not None):
    log1 = vis.Tee(args.outfile, 'w')

# Verbosity levels
# 0: nothing
# 1: only endresults per model
# 2: short summaries and results per epoch
# 3: debug infos (data matrices, examples of data), live progress bar reports
#    per epoch
verbose = args.verbose
if (verbose > 0):
    print("verbosity level: " + str(verbose))
    print(args)


# ---------- Store command line argument variables ----------
# Dry run yes/no?
dry_run = args.dry

# Calculates all possible permutations from the model options selected
# perm = args.perm

# Calculate confusion matrix?
cm = args.cm
do_plot = args.plot

# Train/Test split size
split = args.split

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

if (args.fromfile is not None):
    fromfile = args.fromfile + ".npz"
else:
    fromfile = args.fromfile

# TODO: check arguments for exceptions


# ---------- Data gathering ----------
qry_lmt = args.qry_lmt  # Actual number of posts we will be gathering.

if (args.subreddits is not None):
    subreddit_list = ', '.join("'{0}'".format(s) for s in args.subreddits)
else:
    subreddit_list = pre.subreddits()

max_features = args.max_features  # size of the vocabulary used
seqlen = args.seqlen  # length to which each sentence is padded
maxlen = int(args.maxlen)  # maximum length of comment to be considered
minlen = int(args.minlen)  # minimum length of a comment to be considered
scorerange = args.scorerange
negrange = args.negrange
balanced = args.balanced

if (seqlen > maxlen):
    print("padding length is greater than actual length of the comments.")
    print("setting padding length (" + str(seqlen) + ") to " + str(maxlen))

X_train, X_test, y_train, y_test = get_data(args.dataset,
                                            qry_lmt=qry_lmt,
                                            subreddit_list=subreddit_list,
                                            max_features=max_features,
                                            maxlen=maxlen, minlen=minlen,
                                            scorerange=scorerange,
                                            negrange=negrange,
                                            seqlen=seqlen,
                                            verbose=verbose, split=split,
                                            balanced=balanced,
                                            fromfile=fromfile)

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
if (verbose > 0):
    print("Optimizer permanently set to " + str(opt))
    print("Embedding dim permanently set to " + str(embedding_dim))
    print("Number of filters set to " + str(nb_filter))
    print("Batch size set to " + str(batch_size))
    print("Number of epochs set to " + str(nb_epoch))

# Hieran arbeiten wir heute!

if (len(filter_widths) > 1 or len(dropout_list) > 1 or
        len(activation_list) > 1):
    if (args.model == "parallel"):
        s = [dropout_list, activation_list]
        models = list(product(*s))
    else:
        s = [filter_widths, dropout_list, activation_list]
        models = list(product(*s))
else:
    models = [(filter_widths[0], dropout_p, activation)]

print("Found " + str(len(models)) + " possible models.")
if (query_yes_no("Do you wish to continue?")):
    for m in models:
        if (args.model == "simple"):
            model = CNN_Simple(max_features=max_features,
                               embedding_dim=embedding_dim,
                               seqlen=seqlen,
                               nb_filter=nb_filter,
                               filter_size=m[0],
                               activation=m[2],
                               dropout_p=m[1],
                               l1reg=l1reg, l2reg=l2reg, batchnorm=batchnorm,
                               verbosity=verbose)
        elif (args.model == "twolayer"):
            model = CNN_TwoLayer(max_features=max_features,
                                 embedding_dim=embedding_dim,
                                 seqlen=seqlen,
                                 nb_filter=nb_filter,
                                 filter_size=m[0],
                                 activation=m[2],
                                 dropout_p=m[1],
                                 l1reg=l1reg, l2reg=l2reg, batchnorm=batchnorm,
                                 verbosity=verbose)
        elif (args.model == "parallel"):
            model = CNN_Parallel(max_features=max_features,
                                 embedding_dim=embedding_dim,
                                 seqlen=seqlen,
                                 nb_filter=nb_filter,
                                 filter_widths=filter_widths,
                                 activation=m[1],
                                 dropout_p=m[0],
                                 l1reg=l1reg, l2reg=l2reg, batchnorm=batchnorm,
                                 verbosity=verbose)
        else:
            print("No valid model: " + str(args.model))
        print(model.summary())
        if (dry_run is False):
            model.train(X_train, y_train, X_test, y_test, val=True,
                        opt=opt, nb_epoch=nb_epoch)
            if (cm is True):
                print_cm(model.nn, X_test, y_test)
            if (do_plot is True):
                # We need to change the filenames so this will not be
                # all overwritten..
                o = str(args.outfile)
                vis.plot_nn_graph(model.nn, to_file=o + "-model.png")
                vis.plot_history(model.fitted, to_file=o + "-history.png")
                vis.print_history(model.fitted, to_file=o + "-history.txt")
        print("------------------------------------------------------")

# now we just need a fork/switches for the different models called upon
# by command line and executing the needed plots/prints/fileoutputs
# might as well incude keras functionability to save the compiled and fitted
# model object to disk.

print("======================================================")

# Close logfile
if (args.outfile is not None):
    log1.__del__()
