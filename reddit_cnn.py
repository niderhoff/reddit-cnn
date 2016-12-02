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
    --logreg              calculate logreg benchmark? (default: False)
    --dry                 do not actually calculate anything (default: False)
    --cm                  calculates confusion matrix (default: False)
    --plot                calculate all the plots? (default: False)
    --kfold               0 = regular train/test split according to --split
                          1+ = do k-fold crossvali with this the # of folds
    -v VERBOSE, --verbose VERBOSE
                          verbosity between 0 and 3 (default: 2)
    -f --outfile          filename for plots etc. (default output/datetime.ext)
    -l --logfile          logfile for all output (default: output/datetime.txt)
    --fromfile FROMFILE   file to read datamatrix from (default: None)
                          (currently not in use)

The data are available at
[kaggle](https://www.kaggle.com/reddit/reddit-comments-may-2015).

TODO:

*   also add option to have files named after models if you calculate multiple
*   add option to cut the file-input using qry_lmt
*   possibility to randomize selection from subreddits (as of now it will
    fill all data from first subreddit found if possible)
*   implement non-random initialization for model
*   add docstrings to all functions
*   update existing docstrings
*   add option to time benchmark
*   catch commandline argument exceptions properly
*   add plot with all ROC curves in one. also do better with the distribution
    of ROC/AUC code and actual plot code throughout the files
*   LR code has to be changed to support new code for k-fold crossvalidation.
*   Tables can be printed prittier.
*   Command line Arguments have to be rechecked as some are always on or serve
    no purpose at all right now. (--perm for example)

Known Bugs and Limitations:

*   BATCH NORMALIZATION not working on CUDA v4. (this is an external issue that
    can not be fixed. however, one could think of implementing a check for the
    CUDA version.)
*   SGD optimizer cannot be called through command line (since keras expects
    an actual SGD() call and not a string
*   Newest iteration of database code is really slow.
*   Some documentation is outdated.
*   The current implementation of the CNN-model might not be the best.
*   Right now the log file is always on.
*   Logreg is not the best Benchmark in the world.
*   Writing history to extra file code is outdated and might not work properly.
"""

from __future__ import print_function
import argparse
import sys
import time
import os.path
from itertools import product
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from imblearn.over_sampling import SMOTE

import preprocess as pre
import vis
from models.cnn import CNN_Simple, CNN_TwoLayer, CNN_Parallel
from benchmarks import lr_train, nb_train, svm_train


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
def build_data_matrix(corpus, labels, max_features, seqlen,
                      split, threshold=1):
    X = pre.get_sequences(corpus, max_features, seqlen)
    y = pre.get_labels_binary(labels, threshold)

    if (args.noseed is not True):
        np.random.seed(1234)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=split)
    if (verbose > 1):
        print_data_info(corpus, labels, X_train, X_test, y_train, y_test)
        print('padded example: ' + str(X[1]))
    return (X_train, X_test, y_train, y_test)


def print_data_info(corpus, labels, X_train, X_test, y_train, y_test):
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


# ---------- Undersampling ----------------------
def balanced_subsample(x, y, subsample_size=1.0):
    """
    Courtesy of Charlie Haley.
    From http://stackoverflow.com/questions/23455728/scikit-learn-balanced-
         subsampling
    """

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[np.where(y == yi)]
        class_xs.append((yi, elems))
        if min_elems is None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs, ys


# ---------- Parsing command line arguments ----------
parser = argparse.ArgumentParser(
    description='Reddit CNN - binary classification on reddit comment scores.')

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
parser.add_argument('--threshold', default=1, type=int)
parser.add_argument('--maxlen', default=100, type=int,
                    help='maximum comment length (default: 100)')
parser.add_argument('--minlen', default=0, type=int,
                    help='minimum comment length (default: 0)')
parser.add_argument('--scorerange', nargs=2, default=None, type=int)
parser.add_argument('--negrange', default=False, action='store_true')
parser.add_argument('--balanced', default=None, help="Balance the dataset\
                    by using \'undersampling\' or \'smote\' (default: None)")

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

parser.add_argument('--model', default="simple",
                    help="the type of model (simple/parallel/twolayer)")

# Switches
# For now this switch is always true.
# parser.add_argument('--perm', default=True, action='store_true',
#                     help='calculate all possible model Permutations \
#                     (default: True)')
parser.add_argument('--bench', action='store_true',
                    help='calculate benchmarks? (default: False)')
parser.add_argument('--dry', default=False, action='store_true',
                    help='do not actually calculate anything (default: False)')
parser.add_argument('--cm', default=False, action='store_true',
                    help='calculates confusion matrix (default: False)')
parser.add_argument('--plot', default=False, action='store_true',
                    help='plot the model metrics (default: False)')
parser.add_argument('--kfold', default=0, type=int,
                    help='how many times to perform cross validation (def: 0)')

# Other arguments
out_default = "output/" + time.strftime("%Y%m%d-%H%M%S")
parser.add_argument('-v', '--verbose', default=2, type=int,
                    help='verbosity between 0 and 3 (default: 2)')
parser.add_argument('-f', '--outfile', default=out_default,
                    help='file to output results to (default: None)')
parser.add_argument('-l', '--logfile', default=out_default,
                    help='logfile for all output')
# TODO: add option to disable log file
parser.add_argument('--fromfile', default="store/corpus",
                    help="file input (default: None)")

args = parser.parse_args()


# ---------- Logging ----------
# Initialize logfile if required
if (args.logfile is not None):
    log1 = vis.Tee(args.logfile + ".txt", 'w')

print("Reddit comment data CNN")
print("------------------------------------------------------")

# Verbosity levels
# 0: nothing
# 1: only endresults per model
# 2: short summaries and results per epoch
# 3: debug infos (data matrices, examples of data) and
#    live progress bar reports per epoch
verbose = args.verbose
if (verbose > 0):
    print("verbosity level: " + str(verbose))
    if (verbose > 2):
        print(args)

# TODO: check arguments for exceptions
# Permanent hyperparameters
dropout_p = args.dropout[0]
activation = args.activation[0]
filter_size = args.filters[0]

if (args.fromfile is not None and args.fromfile is not ""):
    args.fromfile = args.fromfile + ".npz"


# ---------- Data gathering ----------
if (verbose > 0):
    print("------------------------------------------------------")
    print("Gathering data.")

if (args.subreddits is not None):
    args.subreddits = ', '.join("'{0}'".format(s) for s in args.subreddits)
else:
    args.subreddits = pre.subreddits()

args.maxlen = int(args.maxlen)  # maximum length of comment to be considered
args.minlen = int(args.minlen)  # minimum length of a comment to be considered

if (args.seqlen > args.maxlen):
    print("padding length is greater than actual length of the comments.")
    print("setting padding length (" +
          str(args.seqlen) + ") to " +
          str(args.maxlen))

if (args.fromfile is not None and os.path.isfile(args.fromfile)):
    f = np.load(args.fromfile)
    raw_corpus, corpus, labels, strata = (f['raw_corpus'], f['corpus'],
                                          f['labels'], f['strata'])
    if (verbose > 1):
        print('Using dataset from file: ' + str(args.fromfile))
else:
    if (verbose > 1):
        print('Using reddit dataset.')
    raw_corpus, corpus, labels, strata = pre.build_corpus(
        args.subreddits,
        args.qry_lmt,
        minlen=args.minlen,
        maxlen=args.maxlen,
        scorerange=args.scorerange,
        negrange=args.negrange,
        batch_size=args.qry_lmt/10,
        verbose=verbose,
        balanced=args.balanced)
print("------------------------------------------------------")
print('corpus length: ' + str(len(corpus)))
print('corpus example: "' + str(corpus[1]) + '"')
print('labels length: ' + str(len(labels)))

# Convert the corpus (i.e. text) into sequences with some Tokenizer magic.
# The sequences will contain numbers referring to a word index table.
X = pre.get_sequences(corpus, args.max_features, args.seqlen)

# The labels (reddit Karma score) (with come in the integer scale) will be
# converted to 0/1 binary values depending on a given threshold.
y = pre.get_labels_binary(labels, args.threshold)
print('labels distribution {}'.format(Counter(y.ravel())))

if (args.balanced == "undersample"):
    X, y = balanced_subsample(X, y)
    print("Rebalancing data. Undersampling has been chosen. New length: " +
          str(X.shape[0]))
    print('Resampled dataset distribution {}'.format(Counter(y)))
elif (args.balanced == "smote"):
    print("Rebalancing data. SMOTE has been chosen.")
    sm = SMOTE()
    X, y = sm.fit_sample(X, y.ravel())
    print('Resampled dataset distribution {}'.format(Counter(y)))

# Validation/Crossvalidation
if (args.kfold > 0):
    # Calculate the splits for k-fold Cross validation. This function actually
    # returns indices, so we will have to get the actual train and test data
    # later on.
    kf = KFold(n_splits=args.kfold)
    # The different folds will be stored in a list to loop over later.
    folds = [[train, test] for train, test in kf.split(X)]
else:
    indices = np.arange(X.shape[0])
    # Regular train/test split without crossvalidation ('0-fold').
    # In this case the data will be split according to a ratio determined with
    # the --split argument.
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(
        X, y, indices, test_size=args.split)
    # To be compatible with the k-fold crossvalidation code used below, we will
    # also get the indices for the data instead of the actual data matrix.
    # The actual split matrices for X and y are basically obsolete and are only
    # used below in the logreg code, but are not really needed. They are here
    # solely for legacy and debugging purposes.
    folds = [[idx1, idx2]]
    if (verbose > 1):
        # Also we need X_train and X_test because this function has not been
        # adapted to work with the k-fold Crossvalidation code.
        print_data_info(corpus, labels, X_train, X_test, y_train, y_test)
        print('padded example: ' + str(X[1]))

print("======================================================")


# ---------- Logistic Regression benchmark ----------
# Run a logistic regression benchmark first so we can later see if our
# ConvNet is somewhere in the ballpark. Still, this is quite a weak benchmark
# and we should think about implement some SVM or Naive Bayes later on.

if (args.bench is True):
    if (verbose > 0):
        print("Running logistic regression benchmarks.")
        print("------------------------------------------------------")
    if (args.dry is False):
        k = len(folds)
        # Scikit-learn logreg.
        # This will also return the predictions to make sure the model doesn't
        # just predict one class only (-> imbalanced dataset problem)

        metrics = {'val': [], 'fpr': [], 'tpr': [], 'roc_auc': [],
                   'predict': []}
        for j in folds:
            X_train, X_test = X[j[0]], X[j[1]]
            y_train, y_test = y[j[0]], y[j[1]]
            # Dataframe to store the metrics, will  be printed later
            predict = lr_train(X_train, y_train, validation_data=(X_test,
                               y_test), verbose=verbose)
            print(predict)
            val = np.mean(predict == y_test)
            fpr, tpr, roc_auc = vis.roc_auc(y_test, predict)
            metrics['val'].append(val)
            metrics['fpr'].append(fpr)
            metrics['tpr'].append(tpr)
            metrics['roc_auc'].append(roc_auc)
            metrics['predict'].append(predict)
        if (verbose > 0):
            print("Validation accuracy avg: " + str(
                  np.mean(metrics['val'][0:2])))
            print('AUC avg: %f' % np.mean(metrics['roc_auc'][0:2]))
        print("------------------------------------------------------")

        # A simple logistic regression from Keras library
        for j in folds:
            X_train, X_test = X[j[0]], X[j[1]]
            y_train, y_test = y[j[0]], y[j[1]]
            model = lr_train(X_train, y_train, validation_data=(X_test,
                             y_test), type='k1', verbose=verbose)
            predict = np.squeeze(model.predict_classes(X_test, verbose=0))
            print(predict)
            val = np.mean(predict == y_test)
            fpr, tpr, roc_auc = vis.roc_auc(y_test, predict)
            metrics['val'].append(val)
            metrics['fpr'].append(fpr)
            metrics['tpr'].append(tpr)
            metrics['roc_auc'].append(roc_auc)
            metrics['predict'].append(predict)
        if (verbose > 0):
            print("Validation accuracy avg: " + str(
                  np.mean(metrics['val'][3:6])))
            print('AUC avg: %f' % np.mean(metrics['roc_auc'][3:6]))
        print("------------------------------------------------------")

        # Keras logreg with l1 and l2 regularization
        for j in folds:
            X_train, X_test = X[j[0]], X[j[1]]
            y_train, y_test = y[j[0]], y[j[1]]
            model = lr_train(X_train, y_train, validation_data=(X_test,
                             y_test), type='k2', verbose=verbose)
            predict = np.squeeze(model.predict_classes(X_test, verbose=0))
            print(predict)
            val = np.mean(predict == y_test)
            fpr, tpr, roc_auc = vis.roc_auc(y_test, predict)
            metrics['val'].append(val)
            metrics['fpr'].append(fpr)
            metrics['tpr'].append(tpr)
            metrics['roc_auc'].append(roc_auc)
            metrics['predict'].append(predict)
        if (verbose > 0):
            print("Validation accuracy avg: " + str(
                  np.mean(metrics['val'][6:9])))
            print('AUC avg: %f' % np.mean(metrics['roc_auc'][6:9]))
    print("======================================================")

# ---------- Naive Bayes benchmark ----------
    if (verbose > 0):
        print("Running Naive Bayes benchmark.")
        print("------------------------------------------------------")
        if (verbose > 1):
            print("Tuning with GridSearchCV and parameters " +
                  "{'alpha': (1e-2, 1e-3, 1e-4), 'fit_prior': (True, False)")
            print("Class priors are adjusted according to data.")
            print("Using default 3-fold Crossvalidation.")
    if (args.dry is "Fale"):
        # gridserach first or cv firsT?
        val, predict = nb_train(X_train, y_train, X_test, y_test)
        fpr, tpr, roc_auc = vis.roc_auc(y_test, predict)
        metrics.append([val, fpr, tpr, roc_auc])
        if (verbose > 0):
            print("\nPredictions: " + str(predict))
            print("Validation accuracy: " + str(val))  # TODO: falsch
            print('AUC: %f' % roc_auc)
    print("======================================================")

# ---------- SVM benchmark ----------
    if (verbose > 0):
        print("Running SVM benchmark (5 epochs).")
        print("------------------------------------------------------")
        if (verbose > 1):
            print("Tuning with GridSearchCV and parameters " +
                  "{'alpha': (1e-2, 1e-3, 1e-4), 'penalty': ('l1', 'l2'," +
                  " 'elasticnet'), 'n_iter': (5, 10), 'l1_ratio': (0.15)}")
            print("Fitting with intercept. 5 epochs, training data shuffled" +
                  " after each epoch.")
            print("Learning rate: eta = 1.0 / (alpha * (t + t0))")
            print("All classes are supposed to have weight 1.")
            print("Using default 3-fold Crossvalidation.")
    if (args.dry is "fals"):
        val, predict = svm_train(X_train, y_train, X_test, y_test)
        fpr, tpr, roc_auc = vis.roc_auc(y_test, predict)
        metrics.append([val, fpr, tpr, roc_auc])
        if (verbose > 0):
            print("\nPredictions: " + str(predict))
            print("Validation accuracy: " + str(val))  # TODO: falsch
            print('AUC: %f' % roc_auc)
    print("======================================================")

# ---------- ANN benchmark ----------
#    if (verbose > 0):
#        print("Running Neural Net benchmark.")
#    if (args.dry is False):
#        nn_train(X_train, y_train, X_test, y_test, args.max_features,
#                 args.embed, args.seqlen, l1reg=0.01, l2reg=0.01)
#    print("======================================================")

    if (args.plot is True):
        # metrics 5x4 list with val, fpr, tpr, roc_auc
        for k in metrics:
            # TODO: we can only do this for 4/5 benchmarks so we have to
            # think of a way to handle this.
            # vis.plot_history(plt.figure(), k[0])
            # This function will plot the ROC curve
            vis.plot_roc(plt.figure(), k[1], k[2], k[3])
        o = str(args.outfile)
        # All plots that have been created so far and are hovering
        # in plotly limbo will now be written to a pdf file.
        vis.multipage(o + "-benchmarks.pdf")
        # Close all plots so they won't appear in next file.
        plt.close('all')


# ---------- Convolutional Neural Network ----------
if (verbose > 0):
    print("Running Convolutonal Neural Network")
    print("------------------------------------------------------")
    print("Optimizer permanently set to " + str(args.opt))
    print("Embedding dim permanently set to " + str(args.embed))
    print("Number of filters set to " + str(args.nb_filter))
    print("Batch size set to " + str(args.batch_size))
    print("Number of epochs set to " + str(args.epochs))

if (len(args.filters) > 1 or len(args.dropout) > 1 or
        len(args.activation) > 1):
    # Here we will check if a selection of more than 1 value is given for
    # any of the main hyperparameters. If this is True, we will proceed to
    # calculate all possible permutations from the given list of hyperparamter
    # values.
    if (args.model == "parallel"):
        # If parallel filter sizes is selected, however, we will only permutate
        # all other hyparmaraters since the different filter sizes will all
        # be used in one model.
        s = [args.dropout, args.activation]
        models = list(product(*s))
    else:
        s = [args.filters, args.dropout, args.activation]
        models = list(product(*s))
        # 'models' now contains tuples with
        #   (dropout percentage, activation function)
else:
    models = [(args.filters[0], dropout_p, activation)]
    # 'models' now contains tuples with
    #   (filter width, dropout percentage, activation function)

print("Found " + str(len(models)) + " possible models to calculate.")
if (query_yes_no("Do you wish to continue?")):
    # For every combination of hyperparameters (m = 1 permuation)
    for m in models:
        # Calculate the selected model architecture.
        # The wording here is very ambiguous. In the above line we refer to a
        # 'model' as a specific combination of hyperparamters regardless of the
        # specific architecture that is used. Below, a model is a specific
        # model architecture (i.e. combination of layers).
        if (args.model == "simple" or args.model == "twolayer"):
            # The 'model'-variable here is the actual application of a certain
            # architecture with a fixed set of hyperparameters. So this is the
            # actual model in the most specific sense.
            parameters = {
                "max_features": args.max_features,
                "embedding_dim": args.embed,
                "seqlen": args.seqlen,
                "nb_filter": args.nb_filter,
                "filter_size": m[0],
                "activation": m[2],
                "dropout_p": m[1],
                "l1reg": args.l1,
                "l2reg": args.l2,
                "batchnorm": args.batchnorm,
                "verbosity": verbose
            }
        elif (args.model == "parallel"):
            parameters = {
                "max_features": args.max_features,
                "embedding_dim": args.embed,
                "seqlen": args.seqlen,
                "nb_filter": args.nb_filter,
                "filter_widths": args.filter_widths,
                "activation": m[1],
                "dropout_p": m[0],
                "l1reg": args.l1,
                "l2reg": args.l2,
                "batchnorm": args.batchnorm,
                "verbosity": verbose
            }
        else:
            print("No valid architecture: " + str(args.model))
        # If there is no --dry switch -> proceed to train actual models.
        if (args.dry is False):
            print("selected " + str(args.kfold) + "-fold (cross-)val")
            # k should be equal to args.kfold in any case :)
            k = len(folds)
            # Create k separate model instances so there is no overlap in
            # information
            if args.model == "simple":
                instances = [CNN_Simple(parameters) for j in range(0, k)]
            elif args.model == "twolayer":
                instances = [CNN_TwoLayer(parameters) for j in range(0, k)]
            elif args.model == "parallel":
                instances = [CNN_Parallel(parameters) for j in range(0, k)]
            # Dataframe to store the metrics, will  be printed later
            metrics = pd.DataFrame(index=range(1, k+1),
                                   columns=('loss', 'val', 'AUC'))
            # List to store all the plots that will be created. In case we
            # create other plots on the side later (not currently implemented)
            # we can use this list to only store the important plots to PDF.
            figs = [plt.figure(j) for j in range(1, 2*k+1)]
            for j in range(1, k+1):
                # Get train and test data for current CV fold and train model.
                X_train, X_test = X[folds[j-1][0]], X[folds[j-1][1]]
                y_train, y_test = y[folds[j-1][0]], y[folds[j-1][1]]
                print("\n")
                print(instances[j-1].summary())
                instances[j-1].train(X_train, y_train, X_test, y_test,
                                     val=True, opt=args.opt,
                                     nb_epoch=args.epochs)
                # Receiver operating characteristic and Area under Curve.
                # Calculate it first and store in a variable, then print
                # the value to stdout.
                y_score = instances[j-1].nn.predict_classes(X_test, verbose=0)
                fpr, tpr, roc_auc = vis.roc_auc(y_test, y_score)
                if (verbose > 0):
                    print('\nAUC: %f' % roc_auc)
                # Save Loss, Validation accuracy, and AUC to the dataframe.
                evalu = instances[j-1].nn.evaluate(X_test, y_test, verbose=0)
                metrics.loc[j, ] = (evalu[0], evalu[1], roc_auc)
                # Print Confusion Matrix if wanted
                if (args.cm is True):
                    vis.print_cm(instances[j-1].nn, X_test, y_test)
                # Create plots
                if (args.plot is True):
                    # This function will print validation and loss over epochs
                    vis.plot_history(figs[j*2-2], instances[j-1].fitted)
                    # This function will plot the ROC curve
                    vis.plot_roc(figs[j*2-1], fpr, tpr, roc_auc)
                    # Making sure we have all the plots alternating as well,
                    # so we always have a pair of history plot and auc plot
                    # before the next CV-fold starts.
            # Print table of loss/validation accuracies  and averages
            print("\nLoss and validation accuracies for all folds:\n")
            print(metrics)
            print("\naveraged values ")
            print(np.mean(metrics))
            if (args.plot is True):
                o = str(args.outfile)
                # All plots that have been created so far and are hovering
                # in plotly limbo will now be written to a pdf file.
                vis.multipage(o + "-outputs.pdf")
                # Call keras function to save a graph of the model architecture
                # to a different file
                # vis.plot_nn_graph(model.nn, to_file=o + "-model.png")
                # Possibility to save a table of the validation/loss history
                # over all epochs to a file. This function is outdated and also
                # it is basically the same data as in the graphs that were
                # created above.
                # vis.print_history(model.fitted, to_file=o + "-history.txt")
        print("------------------------------------------------------")
print("======================================================")

# Close logfile
if (args.logfile is not None):
    log1.__del__()
