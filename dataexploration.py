import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

sns.set_style("ticks")
sns.set_color_codes("pastel")
# _, x, y = np.loadtxt("subreddit_wpc.txt", dtype={
#                'names': ('rank', 'name', 'wpc'),
#                'formats': ('i4', 'S64', 'float64')},
#                unpack=True)
data = pd.read_csv('subreddit_wpc.txt', sep="\t", header=0, index_col=0)[::2]
f, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x="subreddit", y="wpc", data=data,
            label="words per comment", color="b")
ax.set(xlabel='Subreddits', ylabel='Average word count per comment')
ax.set_xticklabels(data['subreddit'], rotation=45, ha="right")
sns.despine(right=True, top=True)
meanline = plt.axhline(y=np.mean(data['wpc']), color="red", linewidth=0.5)
red_patch = mpatches.Patch(color='red', label='Average word count for all \
                                               subreddits')
ax.legend(handles=[red_patch])
f.savefig('subreddit_wpc.png', dpi=200, bbox_inches='tight')


corpora = np.load("store/50krnd.npz")
raw_corpus, corpus, labels, strata = (corpora['raw_corpus'],
                                      corpora['corpus'],
                                      corpora['labels'],
                                      corpora['strata'])
f, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-50, 450, 100)
sns.distplot(pd.Series(labels, name="Comment score"), bins=bins, kde=False)
sns.despine(right=True, top=True)
ax.set_yscale('log')
ax.set_ylabel("log Frequency")
ax.set_xlim([-50, 450])
xticks = np.linspace(-50, 450, 11)
ax.set_xticks(xticks)
f.savefig('labels_log.png', dpi=200, bbox_inches='tight')

f, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-20, 60, 80)
sns.distplot(pd.Series(labels, name="Comment score"), bins=bins, kde=False)
ax.set_xlim([-20, 20])
ax.set_ylim([0, 22000])
yticks = np.linspace(0, 22000, 5)
ax.set_yticks(yticks)
ax.set_ylabel("Frequency")
sns.despine(right=True, top=True)
f.savefig('labels_hist.png', dpi=200, bbox_inches='tight')

Counter(labels)
