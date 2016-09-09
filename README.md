Reddit CNN - binary classification on reddit comment scores.

Reads posts and scores from reddit comment database, provide sequence
embeddings for the comment text to feed into various machine learning models.

Provides functions to train logistic regression from scikit-learn (simple) as
well as keras (simple and with l1 and l2 regularization).

Also trains Convolutional Neural Networks (CNN) with varying filter sizes,
filter numbers and optimizers using keras (theano).

Can be supplied with a range of Filter sizes, Dropout Rates, and activation
functions. The program will then calculate either all possible model
combinations (if --perm is supplied as an argument) or simple models each for
the different filter sizes, dropout rates, or activation functions. The --perm
option can be used in conjunction with --parallel to use all supplied filter
sizes in a single model instead of one per model. This is useful as a second
step after the roughly optimal filter size has been identified and the user
wants to add several filter sizes close to the optimum.

Usage:

    $ reddit_cnn.py [-h] [--dataset DATASET] [-q QRY_LMT]
                     [--max_features MAX_FEATURES] [--maxlen MAXLEN]
                     [--minlen MINLEN] [-b BATCH_SIZE] [-o OPT] [-e EPOCHS]
                     [-N NB_FILTER] [-F [FILTERS [FILTERS ...]]]
                     [-A [ACTIVATION [ACTIVATION ...]]]
                     [-D [DROPOUT [DROPOUT ...]]] [-E EMBED] [--perm]
                     [--logreg] [--dry] [--parallel] [--cm] [-v VERBOSE]
                     [-f FILE]

Args:
      -h, --help            show this help message and exit
      --dataset DATASET     dataset to be used (default: 'reddit')
      -q QRY_LMT, --qry_lmt QRY_LMT
                            amount of data to query (default: 10000)
      --max_features MAX_FEATURES
                            size of vocabulary (default: 5000)
      --maxlen MAXLEN       maximum comment length (default: 100)
      --minlen MINLEN       minimum comment length (default: 0)
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            batch size (default: 32)
      -o OPT, --opt OPT     optimizer flag (default: 'rmsprop')
      -e EPOCHS, --epochs EPOCHS
                            number of epochs for models (default: 5)
      -N NB_FILTER, --nb_filter NB_FILTER
                            number of filters for each size (default: 100)
      -F [FILTERS [FILTERS ...]], --filters [FILTERS [FILTERS ...]]
                            filter sizes to be calculated (default: 3)
      -A [ACTIVATION [ACTIVATION ...]], --activation [ACTIVATION [...]]
                            activation functions to use (default: ['relu'])
      -D [DROPOUT [DROPOUT ...]], --dropout [DROPOUT [DROPOUT ...]]
                            dropout percentages (default: [0.25])
      -E EMBED, --embed EMBED
                            embedding dimension (default: 100)
      --perm                calculate all possible model Permutations (default:
                            True)
      --logreg              calculate logreg benchmark? (default: False)
      --dry                 do not actually calculate anything (default: False)
      --parallel            run filter sizes in parallel (default: False)
      --cm                  calculates confusion matrix (default: False)
      -v VERBOSE, --verbose VERBOSE
                            verbosity between 0 and 3 (default: 2)
      -f FILE, --file FILE  file to output to (default: None)

The data are available at
[kaggle](https://www.kaggle.com/reddit/reddit-comments-may-2015).

TODO:

*   actually more than 1 subreddit?
*   init, activation
*   k-fold crossvalidation
*   add docstrings to everything
*   add option to display+print model plot
*   add option to time benchmark
*   catch argument exceptions
