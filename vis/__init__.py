import sys
from os import path, makedirs
import errno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
from keras.utils.visualize_util import plot
from sklearn.metrics import confusion_matrix
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
        if not path.exists(path.dirname(name)):
            try:
                makedirs(path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
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


def plot_history(fig, history, to_file=None):
    # TODO: make plots not overlap.
    ax1 = fig.add_subplot(211)
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
    fig.set_tight_layout(True)
    if (to_file is not None):
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


def plot_roc(fig, roc_auc, fpr, tpr, to_file=None):
    # y_score = nn.predict(X_test)
    # fpr, tpr, _ = roc_curve(y_test, y_score)
    # roc_auc = auc(fpr, tpr)
    ax0 = fig.add_subplot(111)
    ax0.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax0.plot([0, 1], [0, 1], 'k--')
    ax0.set_xlim([0.0, 1.05])
    ax0.set_ylim([0.0, 1.05])
    ax0.set_xlabel('False Positive Rate')
    ax0.set_ylabel('True Positive Rate')
    ax0.set_title('Receiver operating characteristic curve')
    if (to_file is not None):
        plt.savefig(to_file)


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


def multipage(filename, figs=None, dpi=200):
    if not path.exists(path.dirname(filename)):
        try:
            makedirs(path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
