Reddit CNN - binary classification on reddit comment scores.

Reads posts and scores from reddit comment database, provide sequence
embeddings for the comment text to feed into various machine learning models.

The comments can be filtered by subreddit, karma score, length and an option
can be selected to always create a balanced dataset (i.e. pos. and neg. karma
is evenly distributed).

Provides functions to train logistic regression from scikit-learn (simple) as
well as keras (simple and with l1 and l2 regularization).

Also trains Convolutional Neural Networks (CNN) with varying filter sizes,
filter numbers using keras (theano). Options include different
optimizers, l1 and l2 regularization, batch normalization. Will try to run on
GPU if possible.

Can be supplied with a range of Filter sizes, Dropout Rates, and activation
functions. The program will then calculate either all possible model
combinations (if --perm is supplied as an argument) or simple models each for
the different filter sizes, dropout rates, or activation functions. The --perm
option can be used in conjunction with --parallel to use all supplied filter
sizes in a single model instead of one per model. This is useful as a second
step after the roughly optimal filter size has been identified and the user
wants to add several filter sizes close to the optimum.

Usage:

    $ python reddit_cnn.py [-h] [--dataset DATASET] [-q QRY_LMT]
                     [--max_features MAX_FEATURES] [--seqlen SEQLEN]
                     [--maxlen MAXLEN] [--minlen MINLEN]
                     [--scorerange SCORERANGE SCORERANGE] [--negrange]
                     [--balanced] [-b BATCH_SIZE] [-o OPT] [-e EPOCHS]
                     [-N NB_FILTER] [-F [FILTERS [FILTERS ...]]]
                     [-A [ACTIVATION [ACTIVATION ...]]]
                     [-D [DROPOUT [DROPOUT ...]]] [-E EMBED] [-l1 L1] [-l2 L2]
                     [-s SPLIT] [--batchnorm] [--model MODEL] [--perm]
                     [--logreg] [--dry] [--cm] [-v VERBOSE]
                     [--fromfile FROMFILE]

 Args:

     -h, --help            show help message and exit
     -q QRY_LMT, --qry_lmt QRY_LMT
                           amount of data to query (default: 10000)
     --max_features MAX_FEATURES
                           size of vocabulary (default: 5000)
     --seqlen SEQLEN       length to which sequences will be padded (default:
                           100)
     --maxlen MAXLEN       maximum comment length (default: 100)
     --minlen MINLEN       minimum comment length (default: 0)
     --scorerange SCORERANGE SCORERANGE
     --negrange
     --balanced
     -b BATCH_SIZE, --batch_size BATCH_SIZE
                           batch size (default: 32)
     -o OPT, --opt OPT     optimizer flag (default: 'rmsprop')
     -e EPOCHS, --epochs EPOCHS
                           number of epochs for models (default: 5)
     -N NB_FILTER, --nb_filter NB_FILTER
                           number of filters for each size (default: 100)
     -F [FILTERS [FILTERS ...]], --filters [FILTERS [FILTERS ...]]
                           filter sizes to be calculated (default: 3)
     -A [ACTIVATION [ACTIVATION ...]], --activation [ACTIVATION ACTIVATION ...]
                           activation functions to use (default: ['relu'])
     -D [DROPOUT [DROPOUT ...]], --dropout [DROPOUT [DROPOUT ...]]
                           dropout percentages (default: [0.25])
     -E EMBED, --embed EMBED
                           embedding dimension (default: 100)
     -l1 L1                l1 regularization for penultimate layer
     -l2 L2                l2 regularization for penultimate layer
     -s SPLIT, --split SPLIT
                           train/test split ratio (default: 0.1)
     --batchnorm           add Batch Normalization to activations
     --model MODEL         the type of model (simple/parallel/twolayer)
     --logreg              calculate logreg benchmark? (default: False)
     --dry                 do not actually calculate anything (default: False)
     --cm                  calculates confusion matrix (default: False)
     --plot                calculate all the plots? (default: False)
     --kfold               0 = regular train/test split according to --split
                           1+ = do k-fold crossvali with this the # of folds
     -v VERBOSE, --verbose VERBOSE
                           verbosity between 0 and 3 (default: 2)
     -f --outfile          filename for plots etc. (default output/datetime.ext)
     -l --logfile          logfile for all output (default: output/datetime.txt)
     --fromfile FROMFILE   file to read datamatrix from (default: None)
                           (currently not in use)

The data are available at
[kaggle](https://www.kaggle.com/reddit/reddit-comments-may-2015).

TODO:

*   add possibility to input data matrix from file so it must not be reread
    from sqlite database every time (i.e. separate data grab and model build)
*   add possibility to output to file (the results from the model)
*   possibility to randomize selection from subreddits (as of now it will
    fill all data from first subreddit found if possible)
*   implement non-random initialization for model
*   implement k-fold crossvalidation
*   add docstrings to all functions
*   update existing docstrings
*   add option to display+print model plot
*   add option to time benchmark
*   outsource code for model plot, logreg, and time benchmarks
*   catch commandline argument exceptions properly
*   remove old default model code
*   outsource model code to submodule

Known Bugs and Limitations:

*   BATCH NORMALIZATION not working on CUDA v4. (this is an external issue that
    can not be fixed. however, one could think of implementing a check for the
    CUDA version.)
*   --perm switch is currently always true (since the legacy non-permutation
    procedure that is currently implemented is obsolete).
*   SGD optimizer cannot be called through command line (since keras expects
    an actual SGD() call and not a string.
*   Newest iteration of database code is really slow.
*   Some documentation is false or misleading.
*   The current implementation of the CNN-model might not be the best.
