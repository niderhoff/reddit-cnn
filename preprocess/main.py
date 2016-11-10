#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proprocessing and loading data for Reddit CNN.

This will read some posts from reddit database using the list of subreddits
from `subreddits.txt` to memmory and do some data preprocessing/cleaning steps.

Usage:
    See `reddit_cnn.py`

TODO:
    *  Make this an actual nice wrapper
    *  Rewrite data to have true random from getgo
       (start with line by line and see how fast it is.)
    *  Add option to randomize subreddit on each read
"""
import sqlite3
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


def db_conn(database="database.sqlite"):
    """
    Opens the SQlite database.

    Args:
        database: the database file.

    Returns:
        sqlite connection object

    Database scheme:
    CREATE TABLE May2015(
      created_utc INTEGER, ups INTEGER, subreddit_id, link_id, name,
      score_hidden, author_flair_css_class, author_flair_text, subreddit,
      id, removal_reason, gilded int, downs int, archived, author,
      score int, retrieved_on int, body, distinguished, edited,
      controversiality int, parent_id
     );
    """

    return sqlite3.connect(database)


def subreddits(subreddit_file="subreddits.txt"):
    """
    Read a list of subreddits that are supposed to be used from a file.
    This helps narrowing down the data to more frequent/appropriate
    subreddits.

    Args:
        subrredt_file: text file containing the subreddit names

    Returns:
        Comma separated string with the subreddits read from the file
    """

    with open(subreddit_file, 'r') as f:
        l = f.read().splitlines()
        subreddits = ', '.join("'{0}'".format(s) for s in l)
    return subreddits


def clean_comment(comment, replace_numbers=False):
    """
    Some simple regex substitution to clean the data.
    URLs will be replaced with the token URLURL to flag external information.
    """

    urlsub = '[a-zA-Z]+?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|\
              [!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    subs = {
        "'": "",
        "tl;dr": "tldr"
        # "\\[deleted\\]": ""
    }
    if (replace_numbers is True):
        subs["[0-9]"] = "XXXX"
    r = re.sub(urlsub, 'URLURL', comment).lower()
    for s in subs:
        r = re.sub(s, subs[s], r)
    r = re.sub('[^A-Za-z0-9\.\,]+', ' ', r).strip()
    return(r)


def build_corpus(subreddit_list=subreddits(), qry_lmt=10000, batch_size=1000,
                 no_urls=False, no_deleted=False,
                 minlen=None, maxlen=None, scorerange=None, negrange=False,
                 balanced=False, verbose=1, tofile=None):
    """
    Fetch data from the default database and filter out unwanted posts.
    Database is read in batches and resulting corpus is returned. (...)
    """
    db = db_conn()
    c = db.cursor()
    query = "SELECT subreddit, body, score FROM May2015 WHERE"
    if (scorerange is not None):
        query = query + " score"
        if (negrange is True):
            query = query + " NOT"
        query = query + " BETWEEN {0} and {1} AND".format(*scorerange)
    query = query + " subreddit in ({0})".format(subreddit_list)
    if (verbose > 2):
        print(query)
    c.execute(query)

    if (verbose > 0):
        print("Building corpus.")
    raw_corpus, corpus, labels, strata = [], [], [], []
    n_pos = n_neg = 0
    i = 1
    status = True

    # ---- OUTER LOOP ----
    while True:
        if (verbose > 1):
            print(str(i * batch_size) + " comments fetched.")
        current_batch = c.fetchmany(batch_size)
        if not current_batch:
            if (verbose > 1):
                print("No more rows to fetch.")
            break
        # ---------- INNER LOOP ------------
        # rewrite of old append_corpus_until_done(...) function
        for row in current_batch:
            body = row[1]
            score = row[2]
            check = True
            # exception conditions come here. if not met, post is added
            if (no_deleted is True and body == "[deleted]"):
                # remove posts that consist of [deleted], indicating a post
                # that has been deleted on reddit
                check = False
            if (no_urls is True):
                # if a URL is found, remove the post in its entirety
                regexp = re.compile("[a-zA-Z]+?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|\
                [!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
                if regexp.search(body) is not None:
                    check = False
            # Clean the body from non-alphanumeric characters, replace certain
            # tokens (urls, "tl;dr", etc.)
            body = clean_comment(body)
            if (maxlen is not None and len(body.split()) > maxlen):
                # check if number of words is within required range
                check = False
            if (minlen is not None and len(body.split()) < minlen):
                check = False
            if (balanced is True):
                if (score <= 1 and n_neg > n_pos):
                    # post with negative or neutral karma will be added
                    # if number of positive posts is greater than number of
                    # negative karma posts
                    check = False
                elif (score > 1 and n_neg < n_pos):
                    # post with positive karma will only be added if number
                    # of negative posts is greater than number of positive
                    # or equal
                    check = False
            if (check is True):
                strata.append(row[0])
                raw_corpus.append(row[1])
                corpus.append(str(body))
                labels.append(score)
                if (score > 1):
                    n_pos += 1
                else:
                    n_neg += 1
            if (len(corpus) >= qry_lmt):
                # if sufficient number of valid posts is found, break the loop
                status = False
                break
        # ---------- END INNER LOOP ------------
        i += 1
        if (status is False):
            break
    # ---- END OUTER LOOP ----
    if (verbose > 0):
        print("Found " + str(len(corpus)) + " comments valid to your query.")
        print("Done.")
    if (tofile is not None):
        np.savez(tofile, raw_corpus=raw_corpus, corpus=corpus, labels=labels,
                 strata=strata)
        print("Saved corpus data to " + tofile)
    return(raw_corpus, corpus, labels, strata)


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


def get_labels_binary(labels, threshold=1, verbose=2):
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
