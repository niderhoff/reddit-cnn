from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers.core import Dense
from keras.regularizers import l1l2
import numpy as np


# ---------- Diagnostics and Benchmarks ----------
# Logistic Regression
def lr_train(X_train, y_train, val=True, validation_data=None, type='skl',
             nb_epoch=10, reg=l1l2(l1=0.01, l2=0.01), verbose=1):
    X_test, y_test = validation_data
    if (verbose > 2):
        verb = 1
    elif (verbose == 2):
        verb = 2
    else:
        verb = 0
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
                  validation_data=validation_data, verbose=verb)
        return model
    elif (type == 'k2'):
        # logistic regression with L1 and L2 regularization
        model = Sequential()
        model.add(Dense(1, activation='sigmoid', W_regularizer=reg,
                  input_dim=X_train.shape[1]))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, nb_epoch=nb_epoch,
                  validation_data=validation_data, verbose=verb)
        return model


# Naive Bayes Classifier
def nb_train(X_train, y_train, X_test, y_test, verbose=1, cv=3):
    parameters = {'alpha': (1e-2, 1e-3, 1e-4), 'fit_prior': (True, False)}
    clf = MultinomialNB()
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=cv)
    gs_clf.fit(X_train, y_train.ravel())
    predicted = gs_clf.predict(X_test)
    val = np.mean(predicted == y_test)
    return val, predicted


# SVM benchmark
def svm_train(X_train, y_train, X_test, y_test, cv=3):
    parameters = {'alpha': (1e-2, 1e-3, 1e-4),
                  'penalty': ('l1', 'l2', 'elasticnet'),
                  'n_iter': (5, 10)}
    clf = SGDClassifier(loss='hinge')
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=cv)
    gs_clf.fit(X_train, y_train.ravel())
    predicted = gs_clf.predict(X_test)
    val = np.mean(predicted == y_test)
    return val, predicted


# Simple ANN Benchmarks
# def nn_train(X_train, y_train, X_test, y_test, max_features, embedding_dim,
#              seqlen, l1reg, l2reg):
#     model = Sequential()
#     model.add(Embedding(input_dim=max_features,
#               output_dim=embedding_dim,
#               input_length=seqlen))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(300))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=SGD(),
#                   metrics=['accuracy'])
#     model.fit(X_train, y_train, batch_size=32, nb_epoch=10,
#               validation_data=(X_test, y_test),
#               verbose=0)
#     y_score = model.predict(X_test)
#     fpr, tpr, _ = roc_curve(y_test, y_score)
#     roc_auc = auc(fpr, tpr)
#     print('\nAUC: %f' % roc_auc)
#     print(model.evaluate(X_test, y_test))
#     vis.print_cm(model.nn, X_test, y_test)
