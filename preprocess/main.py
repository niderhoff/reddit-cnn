#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proprocessing and loading data for Reddit CNN.

This will read some posts from reddit database using the list of subreddits
from `subreddits.txt` to memmory and do some data preprocessing/cleaning steps.

Usage:
    See `reddit_cnn.py`

TODO:
    *  actually more than 1 subreddit? (LIMIT will only get data from the first
       subreddit in the list).
    *  Make this a nice wrapper with classes etc.
    *  Randomize data read
"""

import sqlite3
import re


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


def db_conn(database="database.sqlite"):
    """
    Opens the SQlite database.

    Args:
        database: the database file.

    Returns:
        sqlite connection object

    Database schema:
    CREATE TABLE May2015(
      created_utc INTEGER, ups INTEGER, subreddit_id, link_id, name,
      score_hidden, author_flair_css_class, author_flair_text, subreddit,
      id, removal_reason, gilded int, downs int, archived, author,
      score int, retrieved_on int, body, distinguished, edited,
      controversiality int, parent_id
     );
    """

    return sqlite3.connect(database)


def db_query(db_conn, subreddit_list, limit):
    """
    ---- This function is legacy and currently not in use. ----

    Executes the SQL query that will selected the necessary data from
    the sqlite database.

    Args:
        db_conn: sqlite database connection object
        subreddit_list: list of subreddits the comments will be from
        limit: maximum number of posts requested

    Returns:
        sqlite data dump

    TODO:
        *   randomize selection properly instead of taking the first x
        *   always from the same subreddit
        *   fix insecure sql substitution
        *   implement minimum length
    """

    sql_qry = "SELECT subreddit, body, score FROM May2015\
               WHERE subreddit in ({0}) \
               LIMIT {1}".format(subreddit_list, limit)
    print("Querying db...")
    return db_conn.execute(sql_qry)


def get_corpora(subreddit_list, qry_lmt=10000, batch_size=1000,
                minlen=5, maxlen=100, minscore=None, maxscore=None,
                no_urls=False, no_deleted=False, replace_numbers=False,
                verbose=1):
    """
    Returns the raw_corpus, corpus, labels, and strata lists from the
    sqlite database.

    Args:
        subreddit_list: List of the subreddits that will be included in queries
        qry_lmt: Requested corpus size. Will stop the loop if reached.
        batch_size: Size of each database request (for performance)
        minlen: minimum words in a comment required
        maxlen: maximum words in a comment allowed
        minscore: minimum required karma of a comment
        maxscore: maximum allowed karma of a comment
        no_urls: if true exclude all posts that contain urls
        no_deleted: if true exclude all posts that are marked as [deleted]
        replace_numbers: if true replace all numbers by single identifier
        verbose: prevent this function from printing anything if 0

    Returns:
        Tuple of raw_corpus, corpus, labels, and strata.
    """

    db = db_conn()
    c = db.cursor()
    c.execute("SELECT subreddit, body, score FROM May2015 \
              WHERE subreddit in ({0})".format(subreddit_list))
    raw_corpus, corpus, labels, strata = [], [], [], []
    i = 1
    if (verbose > 0):
        print("Building corpus.")
    while True:
        if (verbose > 0):
            print(str(i * batch_size) + " comments fetched.")
        rows = c.fetchmany(batch_size)
        if not rows:
            if (verbose > 0):
                print("No more rows to fetch.")
            break
        status, raw_corpus, corpus, labels, strata = append_corpus_until_done(
            rows, qry_lmt, raw_corpus, corpus, labels, strata, minlen, maxlen,
            minscore, maxscore, no_urls, no_deleted, replace_numbers)
        if (status is False):
            break
        i += 1
    if (verbose > 0):
        print("Found " + str(len(corpus)) + " comments valid to your query.")
        print("Done.")
    return(raw_corpus, corpus, labels, strata)


def append_corpus_until_done(current_batch, qry_lmt,
                             raw_corpus, corpus, labels, strata,
                             minlen=20, maxlen=100, minscore=None,
                             maxscore=None, no_urls=False, no_deleted=False,
                             replace_numbers=False):
    """
    Used by get_corpora(...). Will read from the database in batches, checks
    for validity of the comments given some requirements (see args) and add
    lines to the corpora until qry_lmt valid lines have been fetched.

    Args:
        current_batch: current batch read from the sqlite database in the
                       parent function (get_corpora)
        qry_lmt: number of valid collected comments after which to stop
        raw_corpus: the raw_corpus list that will be appended to
        corpus: the corpus list that will be appended to
        labels: the labels list that will be appended to
        strata: the strata list that will be appended to
        minlen: minimum words in a comment required
        maxlen: maximum words in a comment allowed
        minscore: minimum required karma of a comment
        maxscore: maximum allowed karma of a comment
        no_urls: if true exclude all posts that contain urls
        no_deleted: if true exclude all posts that are marked as [deleted]
        replace_numbers: if true replace all numbers by single identifier

    Returns:
        Tuple of the status variable and raw_corpus, corpus, labels, and strata

    Todo:
        *   Code to exclude posts that include urls or are [deleted]
    """
    for row in current_batch:
        # Different checks to see if a post met the criteria
        if (no_urls is True):
            # TODO: code to exclude post entirely
            print("deleted")
        elif (no_deleted is True):
            # TODO: code to exclude posts with [deleted]
            print("deleted")
        else:
            body = clean_comment(row[1])
            check = True
            if (maxscore is not None):
                if (row[2] > maxscore):
                    check = False
            if (minscore is not None):
                if (row[2] < minscore):
                    check = False
            if (len(body.split()) not in range(minlen, maxlen + 1)):
                check = False
            if (check is True):
                # Post has met all the criteria and will be added to corpus
                raw_corpus.append(row[1])
                corpus.append(str(body))
                labels.append(row[2])
                strata.append(row[0])
        if (len(corpus) >= qry_lmt):
            # Sufficient number of valid posts
            status = False
            break
        else:
            status = True
    return (status, raw_corpus, corpus, labels, strata)


def clean_comment(comment, replace_numbers=False):
    """
    Some simple regex substitution to clean the data.
    """

    urlsub = '[a-zA-Z]+?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|\
              [!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    subs = {
        "'": "",
        "tl;dr": "tldr",
        "\\[deleted\\]": ""
    }
    if (replace_numbers is True):
        subs["[0-9]"] = "XXXX"
    r = re.sub(urlsub, ' ', comment).lower()
    for s in subs:
        r = re.sub(s, subs[s], r)
    r = re.sub('[^A-Za-z0-9\.\,]+', ' ', r).strip()
    return(r)
