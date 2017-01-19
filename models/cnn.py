#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l1, l2, l1l2


class CNN_Simple(object):
    def __init__(self, parameters):
        # max_features, embedding_dim, seqlen,
        # nb_filter, filter_size, activation, dropout_p,
        # l1reg, l2reg, batchnorm,
        self.settings = parameters
        self.settings['verbosity'] = 2
        seqlen = self.settings['seqlen']
        filter_size = self.settings['filter_size']
        l1reg = self.settings['l1reg']
        l2reg = self.settings['l2reg']
        self.nn = Sequential()
        self.nn.add(Embedding(input_dim=self.settings['max_features'],
                              output_dim=self.settings['embedding_dim'],
                              input_length=seqlen))
        self.nn.add(Dropout(self.settings['dropout_p']))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(MaxPooling1D(pool_length=seqlen - filter_size + 1))
        self.nn.add(Flatten())
        self.nn.add(Dropout(self.settings['dropout_p']))
        if (l1reg is not None and l1reg is float and l2reg is not
                None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l1l2(l1reg, l2reg))
        elif (l2reg is not None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l2(l2reg))
        elif (l1reg is not None and l1reg is float):
            self.nn.add(Dense(1), W_regularizer=l1(l1reg))
        else:
            self.nn.add(Dense(1))
        if (self.settings['batchnorm'] is True):
            self.nn.add(BatchNormalization())
        self.nn.add(Activation('sigmoid'))

    def summary(self):
        if (self.settings['verbosity'] == 3):
            print(self.nn.summary())
        elif (self.settings['verbosity'] > 0):
            summary = "MF " + str(self.settings['max_features']) + \
                      " | Len " + str(self.settings['seqlen']) + \
                      " | Embed " + str(self.settings['embedding_dim']) + \
                      " | F " + str(self.settings['filter_size']) + \
                      "x" + str(self.settings['nb_filter']) + \
                      " | Drop " + str(self.settings['dropout_p']) + \
                      " | " + self.settings['activation']
            return summary

    def train(self, X_train, y_train, X_test, y_test, val=False,
              opt='rmsprop', nb_epoch=10, batch_size=32):
        validation_data = (X_test, y_test)
        self.settings['batch_size'] = batch_size
        self.settings['nb_epoch'] = nb_epoch
        self.settings['opt'] = opt
        self.settings['validation'] = val
        if (val is True and validation_data is not None):
            self.nn.compile(loss='binary_crossentropy', optimizer=opt,
                            metrics=['accuracy'])
            if (self.settings['verbosity'] is 1):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=0)
                self.nn.evaluate(X_test, y_test, verbose=0)
            elif (self.settings['verbosity'] is 2):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=2)
            elif (self.settings['verbosity'] is 3):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=1)
        else:
            self.nn.compile(loss='binary_crossentropy', optimizer=opt)
            self.fitted = self.nn.fit(X_train, y_train, batch_size=batch_size,
                                      nb_epoch=nb_epoch)


class CNN_TwoLayer(object):
    def __init__(self, parameters):
        # max_features, embedding_dim, seqlen,
        # nb_filter, filter_size, activation, dropout_p,
        # l1reg, l2reg, batchnorm,
        self.settings = parameters
        self.settings['verbosity'] = 2
        seqlen = self.settings['seqlen']
        l1reg = self.settings['l1reg']
        l2reg = self.settings['l2reg']
        self.nn = Sequential()
        self.nn.add(Embedding(input_dim=self.settings['max_features'],
                              output_dim=self.settings['embedding_dim'],
                              input_length=seqlen))
        self.nn.add(Dropout(self.settings['dropout_p']))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(MaxPooling1D(pool_length=2))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(MaxPooling1D(pool_length=1))
        self.nn.add(Flatten())
        self.nn.add(Dropout(self.settings['dropout_p']))
        if (l1reg is not None and l1reg is float and l2reg is not
                None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l1l2(l1reg, l2reg))
        elif (l2reg is not None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l2(l2reg))
        elif (l1reg is not None and l1reg is float):
            self.nn.add(Dense(1), W_regularizer=l1(l1reg))
        else:
            self.nn.add(Dense(1))
        if (self.settings['batchnorm'] is True):
            self.nn.add(BatchNormalization())
        self.nn.add(Activation('sigmoid'))

    def summary(self):
        if (self.settings['verbosity'] == 3):
            print(self.nn.summary())
        elif (self.settings['verbosity'] > 0):
            summary = "MF " + str(self.settings['max_features']) + \
                      " | Len " + str(self.settings['seqlen']) + \
                      " | Embed " + str(self.settings['embedding_dim']) + \
                      " | F " + str(self.settings['filter_size']) + \
                      "x" + str(self.settings['nb_filter']) + \
                      " | Drop " + str(self.settings['dropout_p']) + \
                      " | " + self.settings['activation']
            return summary

    def train(self, X_train, y_train, X_test, y_test, val=False,
              opt='rmsprop', nb_epoch=10, batch_size=32):
        validation_data = (X_test, y_test)
        self.settings['batch_size'] = batch_size
        self.settings['nb_epoch'] = nb_epoch
        self.settings['opt'] = opt
        self.settings['validation'] = val
        if (val is True and validation_data is not None):
            self.nn.compile(loss='binary_crossentropy', optimizer=opt,
                            metrics=['accuracy'])
            if (self.settings['verbosity'] is 1):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=0)
                self.nn.evaluate(X_test, y_test, verbose=0)
            elif (self.settings['verbosity'] is 2):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=2)
            elif (self.settings['verbosity'] is 3):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=1)
        else:
            self.nn.compile(loss='binary_crossentropy', optimizer=opt)
            self.fitted = self.nn.fit(X_train, y_train, batch_size=batch_size,
                                      nb_epoch=nb_epoch)

# old version (obsolete):
# class CNN_TwoLayer(object):
#     # INPUT -> [CONV -> RELU -> POOL]*2 -> FC
#     def __init__(self,  **settings):
#         # max_features, embedding_dim, seqlen,
#         # nb_filter, filter_size, activation, dropout_p,
#         # l1reg, l2reg, batchnorm,
#         self.settings = settings
#         self.settings['verbosity'] = 2
#         seqlen = self.settings['seqlen']
#         l1reg = self.settings['l1reg']
#         l2reg = self.settings['l2reg']
#         self.nn = Sequential()
#         self.nn.add(Embedding(input_dim=self.settings['max_features'],
#                               output_dim=self.settings['embedding_dim'],
#                               input_length=seqlen))
#         self.nn.add(Dropout(self.settings['dropout_p']))
#         self.nn.add(Convolution1D(self.settings['nb_filter'],
#                                   self.settings['filter_size'],
#                                   activation=self.settings['activation']))
#         self.nn.add(MaxPooling1D(pool_length=2))
#         self.nn.add(Convolution1D(self.settings['nb_filter'],
#                                   self.settings['filter_size'],
#                                   activation=self.settings['activation']))
#         self.nn.add(MaxPooling1D(pool_length=2))
#         self.nn.add(Flatten())
#         self.nn.add(Dropout(self.settings['dropout_p']))
#         if (l1reg is not None and l1reg is float and l2reg is not
#                 None and l2reg is float):
#             self.nn.add(Dense(1), W_regularizer=l1l2(l1reg, l2reg))
#         elif (l2reg is not None and l2reg is float):
#             self.nn.add(Dense(1), W_regularizer=l2(l2reg))
#         elif (l1reg is not None and l1reg is float):
#             self.nn.add(Dense(1), W_regularizer=l1(l1reg))
#         else:
#             self.nn.add(Dense(1))
#         if (self.settings['batchnorm'] is True):
#             self.nn.add(BatchNormalization())
#         self.nn.add(Activation('sigmoid'))
#
#     def summary(self):
#         if (self.settings['verbosity'] == 3):
#             print(self.nn.summary())
#         elif (self.settings['verbosity'] > 0):
#             summary = "MF " + str(self.settings['max_features']) + \
#                       " | Len " + str(self.settings['seqlen']) + \
#                       " | Embed " + str(self.settings['embedding_dim']) + \
#                       " | F " + str(self.settings['filter_size']) + \
#                       "x" + str(self.settings['nb_filter']) + \
#                       " | Drop " + str(self.settings['dropout_p']) + \
#                       " | " + self.settings['activation']
#             return summary
#
#     def train(self, X_train, y_train, X_test, y_test, val=False,
#               opt='rmsprop', nb_epoch=10, batch_size=32):
#         validation_data = (X_test, y_test)
#         self.settings['batch_size'] = batch_size
#         self.settings['nb_epoch'] = nb_epoch
#         self.settings['opt'] = opt
#         self.settings['validation'] = val
#         if (val is True and validation_data is not None):
#             self.nn.compile(loss='binary_crossentropy', optimizer=opt,
#                             metrics=['accuracy'])
#             if (self.settings['verbosity'] is 1):
#                 self.fitted = self.nn.fit(X_train, y_train,
#                                           batch_size=batch_size,
#                                           nb_epoch=nb_epoch,
#                                           validation_data=validation_data,
#                                           verbose=0)
#                 self.nn.evaluate(X_test, y_test, verbose=0)
#             elif (self.settings['verbosity'] is 2):
#                 self.fitted = self.nn.fit(X_train, y_train,
#                                           batch_size=batch_size,
#                                           nb_epoch=nb_epoch,
#                                           validation_data=validation_data,
#                                           verbose=2)
#             elif (self.settings['verbosity'] is 3):
#                 self.fitted = self.nn.fit(X_train, y_train,
#                                           batch_size=batch_size,
#                                           nb_epoch=nb_epoch,
#                                           validation_data=validation_data,
#                                           verbose=1)
#         else:
#             self.nn.compile(loss='binary_crossentropy', optimizer=opt)
#            self.fitted = self.nn.fit(X_train, y_train, batch_size=batch_size,
#                                       nb_epoch=nb_epoch)


class CNN_TwoLayer2(object):
    # INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC
    def __init__(self,  **settings):
        # max_features, embedding_dim, seqlen,
        # nb_filter, filter_size, activation, dropout_p,
        # l1reg, l2reg, batchnorm,
        self.settings = settings
        self.settings['verbosity'] = 2
        seqlen = self.settings['seqlen']
        l1reg = self.settings['l1reg']
        l2reg = self.settings['l2reg']
        self.nn = Sequential()
        self.nn.add(Embedding(input_dim=self.settings['max_features'],
                              output_dim=self.settings['embedding_dim'],
                              input_length=seqlen))
        self.nn.add(Dropout(self.settings['dropout_p']))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(MaxPooling1D(pool_length=2))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(MaxPooling1D(pool_length=2))
        self.nn.add(Flatten())
        self.nn.add(Dropout(self.settings['dropout_p']))
        self.nn.add(Dense(10))
        self.nn.add(Activation(self.settings['activation']))
        if (l1reg is not None and l1reg is float and l2reg is not
                None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l1l2(l1reg, l2reg))
        elif (l2reg is not None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l2(l2reg))
        elif (l1reg is not None and l1reg is float):
            self.nn.add(Dense(1), W_regularizer=l1(l1reg))
        else:
            self.nn.add(Dense(1))
        if (self.settings['batchnorm'] is True):
            self.nn.add(BatchNormalization())
        self.nn.add(Activation('sigmoid'))

    def summary(self):
        if (self.settings['verbosity'] == 3):
            print(self.nn.summary())
        elif (self.settings['verbosity'] > 0):
            summary = "MF " + str(self.settings['max_features']) + \
                      " | Len " + str(self.settings['seqlen']) + \
                      " | Embed " + str(self.settings['embedding_dim']) + \
                      " | F " + str(self.settings['filter_size']) + \
                      "x" + str(self.settings['nb_filter']) + \
                      " | Drop " + str(self.settings['dropout_p']) + \
                      " | " + self.settings['activation']
            return summary

    def train(self, X_train, y_train, X_test, y_test, val=False,
              opt='rmsprop', nb_epoch=10, batch_size=32):
        validation_data = (X_test, y_test)
        self.settings['batch_size'] = batch_size
        self.settings['nb_epoch'] = nb_epoch
        self.settings['opt'] = opt
        self.settings['validation'] = val
        if (val is True and validation_data is not None):
            self.nn.compile(loss='binary_crossentropy', optimizer=opt,
                            metrics=['accuracy'])
            if (self.settings['verbosity'] is 1):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=0)
                self.nn.evaluate(X_test, y_test, verbose=0)
            elif (self.settings['verbosity'] is 2):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=2)
            elif (self.settings['verbosity'] is 3):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=1)
        else:
            self.nn.compile(loss='binary_crossentropy', optimizer=opt)
            self.fitted = self.nn.fit(X_train, y_train, batch_size=batch_size,
                                      nb_epoch=nb_epoch)


class CNN_ThreeLayer(object):
    # INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC
    def __init__(self,  **settings):
        # max_features, embedding_dim, seqlen,
        # nb_filter, filter_size, activation, dropout_p,
        # l1reg, l2reg, batchnorm,
        self.settings = settings
        self.settings['verbosity'] = 2
        seqlen = self.settings['seqlen']
        l1reg = self.settings['l1reg']
        l2reg = self.settings['l2reg']
        self.nn = Sequential()
        self.nn.add(Embedding(input_dim=self.settings['max_features'],
                              output_dim=self.settings['embedding_dim'],
                              input_length=seqlen))
        self.nn.add(Dropout(self.settings['dropout_p']))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(MaxPooling1D(pool_length=2))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(MaxPooling1D(pool_length=2))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(Convolution1D(self.settings['nb_filter'],
                                  self.settings['filter_size'],
                                  activation=self.settings['activation']))
        self.nn.add(MaxPooling1D(pool_length=2))
        self.nn.add(Flatten())
        self.nn.add(Dropout(self.settings['dropout_p']))
        self.nn.add(Dense(10))
        self.nn.add(Activation(self.settings['activation']))
        self.nn.add(Dense(10))
        self.nn.add(Activation(self.settings['activation']))
        if (l1reg is not None and l1reg is float and l2reg is not
                None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l1l2(l1reg, l2reg))
        elif (l2reg is not None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l2(l2reg))
        elif (l1reg is not None and l1reg is float):
            self.nn.add(Dense(1), W_regularizer=l1(l1reg))
        else:
            self.nn.add(Dense(1))
        if (self.settings['batchnorm'] is True):
            self.nn.add(BatchNormalization())
        self.nn.add(Activation('sigmoid'))

    def summary(self):
        if (self.settings['verbosity'] == 3):
            print(self.nn.summary())
        elif (self.settings['verbosity'] > 0):
            summary = "MF " + str(self.settings['max_features']) + \
                      " | Len " + str(self.settings['seqlen']) + \
                      " | Embed " + str(self.settings['embedding_dim']) + \
                      " | F " + str(self.settings['filter_size']) + \
                      "x" + str(self.settings['nb_filter']) + \
                      " | Drop " + str(self.settings['dropout_p']) + \
                      " | " + self.settings['activation']
            return summary

    def train(self, X_train, y_train, X_test, y_test, val=False,
              opt='rmsprop', nb_epoch=10, batch_size=32):
        validation_data = (X_test, y_test)
        self.settings['batch_size'] = batch_size
        self.settings['nb_epoch'] = nb_epoch
        self.settings['opt'] = opt
        self.settings['validation'] = val
        if (val is True and validation_data is not None):
            self.nn.compile(loss='binary_crossentropy', optimizer=opt,
                            metrics=['accuracy'])
            if (self.settings['verbosity'] is 1):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=0)
                self.nn.evaluate(X_test, y_test, verbose=0)
            elif (self.settings['verbosity'] is 2):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=2)
            elif (self.settings['verbosity'] is 3):
                self.fitted = self.nn.fit(X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=1)
        else:
            self.nn.compile(loss='binary_crossentropy', optimizer=opt)
            self.fitted = self.nn.fit(X_train, y_train, batch_size=batch_size,
                                      nb_epoch=nb_epoch)


class CNN_Parallel(object):
    def __init__(self,  **settings):
        # max_features, embedding_dim, seqlen,
        # nb_filter, filter_widths, activation, dropout_p,
        # l1reg, l2reg, batchnorm,
        self.settings = settings
        self.settings['verbosity'] = 2
        seqlen = self.settings['seqlen']
        l1reg = self.settings['l1reg']
        l2reg = self.settings['l2reg']
        conv_filters = []
        for n_gram in settings['filter_widths']:
            sequential = Sequential()
            conv_filters.append(sequential)
            sequential.add(Embedding(input_dim=self.settings['max_features'],
                                     output_dim=self.settings['embedding_dim'],
                                     input_length=seqlen))
            sequential.add(Dropout(self.settings['dropout_p']))
            sequential.add(Convolution1D(self.settings['nb_filter'],
                                         n_gram,
                                         activation=self.settings['activation']
                                         ))
            sequential.add(MaxPooling1D(pool_length=seqlen - n_gram + 1))
            sequential.add(Flatten())
        self.nn = Sequential()
        self.nn.add(Merge(conv_filters, mode='concat'))
        self.nn.add(Dropout(self.settings['dropout_p']))
        if (l1reg is not None and l1reg is float and l2reg is not
                None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l1l2(l1reg, l2reg))
        elif (l2reg is not None and l2reg is float):
            self.nn.add(Dense(1), W_regularizer=l2(l2reg))
        elif (l1reg is not None and l1reg is float):
            self.nn.add(Dense(1), W_regularizer=l1(l1reg))
        else:
            self.nn.add(Dense(1))
        if (self.settings['batchnorm'] is True):
            self.nn.add(BatchNormalization())
        self.nn.add(Activation('sigmoid'))

    def summary(self):
        if (self.settings['verbosity'] == 3):
            print(self.nn.summary())
        elif (self.settings['verbosity'] > 0):
            summary = "MF " + str(self.settings['max_features']) + \
                      " | Len " + str(self.settings['seqlen']) + \
                      " | Embed " + str(self.settings['embedding_dim']) + \
                      " | F " + str(self.settings['filter_widths']) + \
                      "x" + str(self.settings['nb_filter']) + \
                      " | Drop " + str(self.settings['dropout_p']) + \
                      " | " + self.settings['activation']
            return summary

    def train(self, X_train, y_train, X_test, y_test, val=False,
              opt='rmsprop', nb_epoch=10, batch_size=32):
        validation_data = (X_test, y_test)
        concat_X_test = []
        concat_X_train = []
        for i in range(len(self.settings['filter_widths'])):
            concat_X_test.append(X_test)
            concat_X_train.append(X_train)
        self.settings['batch_size'] = batch_size
        self.settings['nb_epoch'] = nb_epoch
        self.settings['opt'] = opt
        self.settings['validation'] = val
        if (val is True and validation_data is not None):
            self.nn.compile(loss='binary_crossentropy', optimizer=opt,
                            metrics=['accuracy'])
            if (self.settings['verbosity'] is 1):
                self.fitted = self.nn.fit(concat_X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=0)
                self.nn.evaluate(concat_X_test, y_test, verbose=0)
            elif (self.settings['verbosity'] is 2):
                self.fitted = self.nn.fit(concat_X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=(concat_X_test,
                                                           y_test),
                                          verbose=2)
            elif (self.settings['verbosity'] is 3):
                self.fitted = self.nn.fit(concat_X_train, y_train,
                                          batch_size=batch_size,
                                          nb_epoch=nb_epoch,
                                          validation_data=validation_data,
                                          verbose=1)
        else:
            self.nn.compile(loss='binary_crossentropy', optimizer=opt)
            self.fitted = self.nn.fit(concat_X_train, y_train,
                                      batch_size=batch_size,
                                      nb_epoch=nb_epoch)
