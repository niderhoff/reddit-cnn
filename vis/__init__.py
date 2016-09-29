import sys
import matplotlib.pyplot as plt
from astropy.table import Table
from keras.utils.visualize_util import plot


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


def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def print_history(history, to_file):
    log1 = Tee(to_file, 'a')
    print(Table(history.history))
    log1.__del__()


def plot_nn_graph(nnobj, to_file=False,
                  show_shapes=False, show_layer_names=False):
    plot(nnobj, to_file=to_file, show_shapes=show_shapes,
         show_layer_names=show_layer_names)
