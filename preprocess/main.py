#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proprocessing and loading data for Reddit CNN.

This will read some posts from reddit database using the list of subreddits
from `subreddits.txt` to memmory and do some data preprocessing/cleaning steps.

Usage:
    See `reddit_cnn.py`

TODO:
    *  data cleaning, this is _actually_ important
    *  actually more than 1 subreddit? (LIMIT will only get data from the first
       subreddit in the list).
    *  Make this a nice wrapper with classes etc.
    *  Randomize data read
"""

import sqlite3
import re


def subreddits(subreddit_file="subreddits.txt"):
    # Read a list of subreddits that are supposed to be used from a file.
    # This helps narrowing down the data to more frequent/appropriate
    # subreddits.
    with open(subreddit_file, 'r') as f:
        l = f.read().splitlines()
        subreddits = ', '.join("'{0}'".format(s) for s in l)
    return subreddits


def db_conn(database="database.sqlite"):
    # Database schema:
    # CREATE TABLE May2015(
    #  created_utc INTEGER, ups INTEGER, subreddit_id, link_id, name,
    #  score_hidden, author_flair_css_class, author_flair_text, subreddit,
    #  id, removal_reason, gilded int, downs int, archived, author,
    #  score int, retrieved_on int, body, distinguished, edited,
    #  controversiality int, parent_id
    # );
    return sqlite3.connect(database)


def db_query(db_conn, subreddit_list, limit):
    # TODO: randomize selection properly instead of taking the first x
    #       always from the same subreddit
    # TODO: fix insecure sql substitution
    sql_qry = "SELECT subreddit, body, score FROM May2015\
               WHERE subreddit in (%s)\
               LIMIT " % subreddit_list + str(limit)
    print("Querying db...")
    return db_conn.execute(sql_qry)


def get_corpus(data):
    print("Building corpus...")
    raw_corpus, corpus, labels, strata = [], [], [], []

    # TODO: insert proper data cleaning routine here
    #   *  strip [deleted]
    for p in data:
        raw_corpus.append(re.sub('\n', '', p[1]))
        cln_post = re.sub('[^A-Za-z0-9\.\,]+', ' ', p[1]).strip().lower()
        corpus.append(str(cln_post))
        labels.append(p[2])
        strata.append(p[0])

    return raw_corpus, corpus, labels, strata
