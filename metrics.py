# benchmark metrics

# from itertools import product
import numpy as np
from tabulate import tabulate

# d_lens = ["10k", "30k", "50k", "100k"]
# d_mod = ["all", "rnd"]
# d_types = ["maxlen",
#            "minlen_ran-12", "minlen_ran-55", "minlen_ran03",
#            "minlen",
#            "minlen5_ran-55", "minlen5_ran03", "minlen5_ran-12",
#            "minlen5",
#            "minmax",
#            "minmax_ran-55", "minmax_ran03", "minmax_ran-12",
#            "minmax5",
#            "ran-12", "ran03", "ran-55"]
#
# names = zip(list(product(*[d_lens, d_mod, d_types])))

# w = {
#     'names': [
#         "50krnd_minlen.npz",
#         "50krnd_minlen5_ran-55.npz",
#         "50krnd_minlen5_ran03.npz",
#         "50krnd_minlen5.npz",
#         "50krnd_minmax_ran-12.npz"
#     ],
#     'files': [
#         "20170113-141209",
#         "20170113-141446",
#         "20170113-142333",
#         "20170113-143017",
#         "20170113-143309"
#     ]
# }

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
        "20170014-144154",
        "20170014-141341",
        "20170014-140306",
        "20170114-134905"
    ]
}

w_minlen_50k = {
    'names': [
        "50krnd_minlen_ran-12",
        "50krnd_minlen_ran03",
        "50krnd_minlen_ran-555"
    ],
    'files': [
        ""
    ]
}


def create_table(w):
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
        headers=['Sample', 'LR1', 'LR2', 'LR3', 'NB', 'SVM']
    ))
    table = zip(w['names'], auc_skl_m, auc_k1_m, auc_k2_m, auc_nb, auc_svm)
    print(tabulate(
        table,
        tablefmt="latex_booktabs",
        headers=['Sample', 'LR1', 'LR2', 'LR3', 'NB', 'SVM']
    ))

create_table(w_size_rnd)

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
