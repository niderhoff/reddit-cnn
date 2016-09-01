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
