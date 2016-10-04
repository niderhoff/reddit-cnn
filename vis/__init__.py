import sys
import matplotlib.pyplot as plt
from astropy.table import Table
from keras.utils.visualize_util import plot
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
