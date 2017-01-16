# benchmark metrics

# from itertools import product
import numpy as np
from tabulate import tabulate

# first by sample size
w_size_rnd = {
    'names': [
        "10krnd",
        "30krnd",
        "50krnd",
        "100krnd"
    ],
    'files': [
        "20170113-114844",
        "20170113-130314",
        "20170113-154800",
        "20170114-133631"
    ]
}

# same for 1 subreddit (android)
w_size_all = {
    'names': [
        "10kall",
        "30kall",
        "50kall",
        "100kall"
    ],
    'files': [
        "20170114-144154",
        "20170114-141341",
        "20170114-140306",
        "20170114-134905"
    ]
}

# -----------------------------------------------------

# min length effect
w_minmax_mix_rnd = {
    'names': [
        "50krnd_minlen",
        "50krnd_maxlen",
        "50krnd_minmax",
        "50kall_minlen",
        "50kall_maxlen",
        "50kall_minmax"
    ],
    'files': [
        "20170113-141209",
        "20170113-130509",
        "20170113-150447",
        "20170114-170906",
        "20170114-172358",
        "20170114-161409"
    ]
}

w_minlen_rnd = {
    'names': [
        "10krnd_minlen",
        "30krnd_minlen",
        "30krnd_minlen (SMOTE)",
        "50krnd_minlen",
        "100krnd_minlen"
    ],
    'files': [
        "20170113-115212",
        "20170113-121330",
        "20170113-121543",
        "20170113-141209",
        "20170114-180138"
    ]
}

w_minlen_all = {
    'names': [
        "10kall_minlen",
        "10kall_maxlen",
        "30kall_minlen",
        "30kall_maxlen",
        "50kall_minlen",
        "50kall_maxlen",
        "100kall_minlen",
        "100kall_maxlen"
    ],
    'files': [
        "20170116-201214",
        "20170116-201440",
        "20170116-201440",
        "20170116-202045",
        "20170114-170906",
        "20170114-172358",
        "20170113-190343",
        "20170113-160702"
    ]
}

# -----------------------------------------------------


# range effect
w_ran_mix = {
    'names': [
        "50krnd_ran-12*",
        "50krnd_ran03",
        "100kall_ran-12",
        "100kall_ran03",
    ],
    'files': [
        "20170113-151134",
        "20170113-153943",
        "20170114-181604",
        "20170114-185646"
    ]
}

# -----------------------------------------------------


def create_table_cnn(w):
    val_cnn_m, val_cnn_v = [], []
    auc_cnn_m, auc_cnn_v = [], []
    for npz in w['files']:
        f = np.load("output/" + npz + "-model.npz")
        cnn_metrics = f['arr_1'][0]
        val_cnn_m.append(np.mean(cnn_metrics)[1])
        val_cnn_v.append(np.var(cnn_metrics)[1])
        auc_cnn_m.append(np.mean(cnn_metrics)[2])
        auc_cnn_v.append(np.var(cnn_metrics)[2])
    table = zip(w['names'], val_cnn_m, val_cnn_v, auc_cnn_m, auc_cnn_v)
    print(tabulate(
        table,
        tablefmt="latex_booktabs",
        floatfmt=".3f",
        headers=['Sample', 'Mean Acc.', 'Var. Acc.', 'Mean AUC', 'Var. AUC']
    ))


def create_table_bench(w):
    val_skl_m, val_skl_v, val_k1_m, val_k1_v, val_k2_m, val_k2_v, val_nb, \
        val_svm = [], [], [], [], [], [], [], []
    auc_skl_m, auc_k1_m, auc_k2_m, auc_nb, auc_svm = [], [], [], [], []
    auc_skl_v, auc_k1_v, auc_k2_v = [], [], []
    for npz in w['files']:
        f = np.load("output/" + npz + "-bench.npz")
        lr_metrics = f['arr_0'].item()
        svm_metrics = f['arr_2'].item()
        nb_metrics = f['arr_1'].item()
        # lr method sklearn
        val_skl_m.append(np.mean(lr_metrics['val'][0:10]))
        val_skl_v.append(np.var(lr_metrics['val'][0:10]))
        auc_skl_m.append(np.mean(lr_metrics['roc_auc'][0:10]))
        auc_skl_v.append(np.var(lr_metrics['roc_auc'][0:10]))
        # lr method keras1
        val_k1_m.append(np.mean(lr_metrics['val'][10:20]))
        val_k1_v.append(np.var(lr_metrics['val'][10:20]))
        auc_k1_m.append(np.mean(lr_metrics['roc_auc'][0:10]))
        auc_k1_v.append(np.var(lr_metrics['roc_auc'][0:10]))
        # lr method keras2
        val_k2_m.append(np.mean(lr_metrics['val'][20:30]))
        val_k2_v.append(np.var(lr_metrics['val'][20:30]))
        auc_k2_m.append(np.mean(lr_metrics['roc_auc'][0:10]))
        auc_k2_v.append(np.var(lr_metrics['roc_auc'][0:10]))
        # NB
        val_nb.append(nb_metrics['val'][1])
        auc_nb.append(nb_metrics['roc_auc'][0])
        # SVM
        val_svm.append(svm_metrics['val'][1])
        auc_svm.append(svm_metrics['roc_auc'][0])
    table = zip(w['names'], val_skl_m, val_k1_m, val_k2_m, val_nb, val_svm)
    # add: varianz in klammern (GEIL!)
    print(tabulate(
        table,
        tablefmt="latex_booktabs",
        floatfmt=".3f",
        headers=['Sample', 'LR1', 'LR2', 'LR3', 'NB', 'SVM']
    ))
    table = zip(w['names'], auc_skl_m, auc_k1_m, auc_k2_m, auc_nb, auc_svm)
    print(tabulate(
        table,
        tablefmt="latex_booktabs",
        floatfmt=".3f",
        headers=['Sample', 'LR1', 'LR2', 'LR3', 'NB', 'SVM']
    ))

# create_table_bench(w_size_all)
# create_table_bench(w_size_rnd)
# create_table_bench(w_minlen_rnd)
# create_table_bench(w_minmax_mix_rnd)
create_table_bench(w_minlen_all)
# create_table_bench(w_ran_mix)


# NB metrics
# # calculate fold average here
# val = nb_metrics['val'][1]  # average??
# fpr = nb_metrics['fpr'][0]
# tpr = nb_metrics['tpr'][0]
# roc_auc = nb_metrics['roc_auc'][0]
#
# # fold results
# cvresults = nb_metrics['val'][0]  # folds
# col_no = np.array(range(0, len(cvresults))).astype(int)
# table = {
#     'folds': col_no, 'validation accuray': np.round(cvresults, 3)
# }
#
# print(tabulate(
#     table,
#     tablefmt="latex_booktabs",
#     # floatfmt=".3f",
#     headers="keys")
# )
