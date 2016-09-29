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
        return model.fit(concat_X_train, y_train, batch_size=batch_size,
                         nb_epoch=nb_epoch,
                         validation_data=(concat_X_test, y_test), verbose=0)
        print(model.evaluate(concat_X_test, y_test, verbose=0))
    elif (verbose is 2):
        return model.fit(concat_X_train, y_train, batch_size=batch_size,
                         nb_epoch=nb_epoch,
                         validation_data=(concat_X_test, y_test), verbose=2)
    elif (verbose is 3):
        return model.fit(concat_X_train, y_train, batch_size=batch_size,
                         nb_epoch=nb_epoch,
                         validation_data=(concat_X_test, y_test), verbose=1)
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model.fit(concat_X_train, y_train, batch_size=batch_size,
                         nb_epoch=nb_epoch)
