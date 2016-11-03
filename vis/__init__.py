import sys
import matplotlib.pyplot as plt
from astropy.table import Table
from keras.utils.visualize_util import plot
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np


class Tee(object):
    """
    This is a basically an inside-python rewrite of the unix program 'tee'.
    Will output to stdout and to a file simultaneously if log class is called
    until __del__() is called.

    Doing this from within python will make this platform-agnostic (i.e. work
    in windows, too, without installing a version of tee there manually.)
    """
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        pass


def plot_history(history, to_file):
    # TODO: make plots not overlap.
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')
    ax2 = fig.add_subplot(212)
    # summarize history for loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    plt.savefig(to_file)


def print_history(history, to_file):
    np.set_printoptions(threshold=np.inf)
    log1 = Tee(to_file, 'a')
    print(Table(history.history))
    log1.__del__()


def plot_nn_graph(nnobj, to_file=False,
                  show_shapes=False, show_layer_names=False):
    plot(nnobj, to_file=to_file, show_shapes=show_shapes,
         show_layer_names=show_layer_names)


def plot_roc(model, X_test, y_test, y_score):
    y_score = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    print('AUC: %f' % roc_auc)


def print_cm(nn, X_test, y_test, batch_size=32):
    print("Confusion Matrix (frequency, normalized):")
    y_pred = nn.predict_classes(X_test, batch_size=batch_size, verbose=0)
    print(y_pred)
    print(sum(y_pred)/len(y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    np.set_printoptions(precision=3, suppress=True)
    print(cm)
    print(cm_normalized)
