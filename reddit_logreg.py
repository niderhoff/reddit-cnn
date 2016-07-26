'''
    Logistic regression linear classifier as benchmark for the CNN.
'''

from __future__ import print_function
import sqlite3
import re
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV


np.random.seed(2222)

qry_lmt = 25000
vocab_size = 5000
embedding_dims = 100
paddedlength = 100  # length to which each sentence is padded
batch_size = 32
epochs = 5            # number of training epochs


print("Querying db...")

with open('subreddits.txt', 'r') as f:
    l = f.read().splitlines()
    subreddits = ', '.join("'{0}'".format(s) for s in l)

sql_conn = sqlite3.connect("database.sqlite")

# TODO: actually more than 1 subreddit?
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

lr = LogisticRegressionCV()
lr.fit(X_train, Y_train)
pred_y = lr.predict(X_test)
print("Test fraction correct (LR-Accuracy) = {:.2f}".format(lr.score(X_test,
                                                                     Y_test)))
# 89%

# model = Sequential()
# model.add(Dense(32, input_shape=(20000, 100), W_regularizer=l2(0.01)))
# model.add(Activation('sigmoid'))
#
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='binary_crossentropy', optimizer=sgd)
#
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
#           validation_data=(X_test, Y_test))
# results = model.evaluate(X_test, Y_test, verbose=0)
#
# print(results)
