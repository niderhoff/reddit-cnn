'''
    This will read some posts from reddit database and train a convolutional
    neural network on them with 1 convolutional layer using keras/theano.
    So far, this is just using the imdb.py example from keras and inputting
    the reddit data, but will update the model architecture to fit the data
    better soon.
'''
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
from keras.utils.visualize_util import plot


# TODO: better cleaning, this is _actually_ important

# Parameters
qry_lmt = 10000
vocab_size = 5000
embedding_dims = 100
paddedlength = 100  # length to which each sentence is padded
num_filters = 25  # number of filters to apply/learn in the 1D convolutional
filter_length = 2     # linear length of each filter (this is 1D)

hidden_dimsa = 250    # number of output neurons for the first Dense layer
hidden_dimsb = 100    # number of output neurons for the second Dense layer

batch_size = 32
epochs = 5            # number of training epochs

# Getting the data

print("Querying db...")

with open('subreddits.txt', 'r') as f:
    l = f.read().splitlines()
    subreddits = ', '.join("'{0}'".format(s) for s in l)

sql_conn = sqlite3.connect("database.sqlite")

some_data = sql_conn.execute("SELECT subreddit, body, score FROM May2015\
                              WHERE subreddit IN (%s)\
                              LIMIT " % subreddits + str(qry_lmt))

print("Building corpus...")
raw_corpus, corpus, labels, strata = [], [], [], []

for post in some_data:
    raw_corpus.append(re.sub('\n', '', post[1]))
    cln_post = re.sub('[^A-Za-z0-9\.\,]+', ' ', post[1]).strip().lower()
    corpus.append(str(cln_post))
    labels.append(post[2])
    strata.append(post[0])

# Building the model
print("Creating train/test split")
tokenizer = Tokenizer(nb_words=vocab_size)
tokenizer.fit_on_texts(corpus)
seq = tokenizer.texts_to_sequences(corpus)
X = sequence.pad_sequences(seq, maxlen=paddedlength)
Y = np.asarray(labels)
Y = Y > 0
Y.astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape)

# Embedding
nn = Sequential()
# TODO: weights initialize
nn.add(Embedding(input_dim=vocab_size, output_dim=embedding_dims,
                 input_length=paddedlength))
# TODO: dropout layer
nn.add(Dropout(0.5))

# Convolutions TODO: init, activation
nn.add(Convolution1D(num_filters, filter_length, activation="relu"))

nn.add(MaxPooling1D(pool_length=2))

nn.add(Flatten())

nn.add(Dense(hidden_dimsa))
nn.add(Dropout(0.25))
nn.add(Activation('relu'))

nn.add(Dense(hidden_dimsa))
nn.add(Dropout(0.25))
nn.add(Activation('relu'))

nn.add(Dense(1))
nn.add(Activation('sigmoid'))

nn.compile(loss='binary_crossentropy', optimizer='adam')
plot(nn, to_file='model.png')

nn.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
       validation_data=(X_test, Y_test))
results = nn.evaluate(X_test, Y_test, verbose=0)
print(results)
