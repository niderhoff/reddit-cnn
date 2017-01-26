# benchmark metrics script (not part of the actual code for the CNN)
# it is merely used to produce some tables used in the paper
# this file is work in progress

# from itertools import product
import numpy as np
from tabulate import tabulate
import pandas as pd

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

# auch mit RND und vergleichen danke.
w_all_ran_03 = {
    'names': [
        "10kall_ran03",
        "10kall_ran03 SMOTE",
        "30kall_ran03",
        "30kall_ran03 SMOTE",
        "50kall_ran03",
        "50kall_ran03 SMOTE",
        "100kall_ran03",
        "100kall_ran03 SMOTE"
    ],
    'files': [
        "20170124-115625",
        "20170124-120011",
        "20170124-122817",
        "20170124-123402",
        "20170124-123745",
        "20170124-124624",
        "20170114-185646",
        "20170124-125647"
    ]
}

# -----------------------------------------------------
# CNN stuff
index = ['30krnd', '30kall',
         '50krnd', '50kall',
         '100krnd', '100kall']
w_filters = {
    '3': ["20170124-152930", "20170125-141240",
          "20170124-204810", "20170125-115709",
          "20170124-205912", "20170125-113848"],
    '4': ["20170124-153542", "20170125-141240",
          "20170124-203842", "20170125-125420",
          "20170124-211632", "20170125-111828"],
    '5': ["20170124-162517", "20170125-150258",
          "20170124-202837", "20170125-130720",
          "20170124-213441", "20170125-110028"],
    '6': ["20170124-163247", "20170125-140031",
          "20170124-200516", "20170125-132213",
          "20170124-215817", "20170125-103240"],
    '7': ["20170124-171355", "20170125-135329",
          "20170124-194915", "20170125-134211",
          "20170125-143941", "20170125-100725"]
}
index_minlen = ['30kall minlen', '50kall minlen', '100kall minlen']
w_filters_minlen = {
    '3': ["20170125-141925", "20170125-175623", "20170125-173531"],
    '4': ["20170125-151256", "20170125-181841", "20170125-170955"],
    '5': ["20170125-152114", "20170125-183529", "20170125-165212"],
    '6': ["20170125-153758", "20170125-184440", "20170125-163327"],
    '7': ["20170125-154552", "20170125-155511", "20170125-160925"]
}

index_minlen_ran = ['100kall 03', '100kall 03 smote',
                    '100kall minlen 03', '100kall minlen 03 smote']
w_filters_minlen03 = {
    '3': ["20170126-111259", "", "20170125-201814", "20170125-205028"],
    '7': ["20170126-105331", "", "20170125-193923", "20170125-212013"],
}
# -----------------------------------------------------


def create_table_filter_sizes(w, index):
    d_val = {}
    d_auc = {}
    for i in w:
        ival = []
        iauc = []
        for npz in w[i]:
            if (npz != ""):
                f = np.load("output/" + npz + "-model.npz")
                cnn_metrics = f['arr_1']
                ival.append(np.nanmean(cnn_metrics, axis=0)[1])
                iauc.append(np.nanmean(cnn_metrics, axis=0)[2])
            else:
                ival.append("NaN")
                iauc.append("NaN")
        d_val[i] = ival
        d_auc[i] = iauc
    d_val = pd.DataFrame(d_val, index=index)
    d_auc = pd.DataFrame(d_auc, index=index)
    print(tabulate(d_val,
                   tablefmt="latex_booktabs",
                   floatfmt=".3f",
                   showindex=True,
                   headers="keys"))
    print(tabulate(d_auc,
                   tablefmt="latex_booktabs",
                   floatfmt=".3f",
                   showindex=True,
                   headers="keys"))


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
        val_skl_m.append(np.nanmean(lr_metrics['val'][0:10]))
        val_skl_v.append(np.var(lr_metrics['val'][0:10]))
        auc_skl_m.append(np.nanmean(lr_metrics['roc_auc'][0:10]))
        auc_skl_v.append(np.var(lr_metrics['roc_auc'][0:10]))
        # lr method keras1
        val_k1_m.append(np.nanmean(lr_metrics['val'][10:20]))
        val_k1_v.append(np.var(lr_metrics['val'][10:20]))
        auc_k1_m.append(np.nanmean(lr_metrics['roc_auc'][0:10]))
        auc_k1_v.append(np.var(lr_metrics['roc_auc'][0:10]))
        # lr method keras2
        val_k2_m.append(np.nanmean(lr_metrics['val'][20:30]))
        val_k2_v.append(np.var(lr_metrics['val'][20:30]))
        auc_k2_m.append(np.nanmean(lr_metrics['roc_auc'][0:10]))
        auc_k2_v.append(np.var(lr_metrics['roc_auc'][0:10]))
        # NB
        val_nb.append(nb_metrics['val'][1])
        auc_nb.append(nb_metrics['roc_auc'][0])
        # SVM
        val_svm.append(svm_metrics['val'][1])
        auc_svm.append(svm_metrics['roc_auc'][0])
    table = zip(w['names'], val_skl_m, val_k1_m, val_k2_m, val_nb, val_svm)
    # add: varianz in klammern(?)
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
# create_table_bench(w_minlen_all)
# create_table_bench(w_ran_mix)
# create_table_bench(w_all_ran_03)

#create_table_filter_sizes(w_filters, index)
create_table_filter_sizes(w_filters_minlen, index_minlen)

# old stuff
# def create_table_cnn(w):
#     val_cnn_m, val_cnn_v = [], []
#     auc_cnn_m, auc_cnn_v = [], []
#     for npz in w['files']:
#         f = np.load("output/" + npz + "-model.npz")
#         cnn_metrics = f['arr_1'][0]
#         val_cnn_m.append(np.nanmean(cnn_metrics)[1])
#         val_cnn_v.append(np.var(cnn_metrics)[1])
#         auc_cnn_m.append(np.nanmean(cnn_metrics)[2])
#         auc_cnn_v.append(np.var(cnn_metrics)[2])
#     table = zip(w['names'], val_cnn_m, val_cnn_v, auc_cnn_m, auc_cnn_v)
#     print(tabulate(
#         table,
#         tablefmt="latex_booktabs",
#         floatfmt=".3f",
#         headers=['Sample', 'Mean Acc.', 'Var. Acc.', 'Mean AUC', 'Var. AUC']
#     ))

# NB metrics
# # calculate cv fold average here
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
