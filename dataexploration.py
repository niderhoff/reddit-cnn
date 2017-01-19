import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_style("whitegrid")
sns.set_style("ticks")

# _, x, y = np.loadtxt("subreddit_wpc.txt", dtype={
#                'names': ('rank', 'name', 'wpc'),
#                'formats': ('i4', 'S64', 'float64')},
#                unpack=True)
data = pd.read_csv('subreddit_wpc.txt', sep="\t", header=0, index_col=0)[::2]
f, ax = plt.subplots(figsize=(12, 6))
sns.set_color_codes("pastel")
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
