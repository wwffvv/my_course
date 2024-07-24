import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# file='GRU'
# data = np.load(f'{file}.npz', allow_pickle=True)
# d = {'loss': data['losses'][::100], 'iter':list(range(1, len(data['losses'])+1))[::100]}

# print(file, ":")
# print(data['score'])

filenames = ['GRU', 'LSTM', 'BiLSTM', 'dot_attn', 'transformer']
for file in filenames:
    data = np.load(f'{file}.npz', allow_pickle=True)
    print("***************", file ,"*******************")
    print(data['score'])



# df = pd.DataFrame(d)
# plt.figure(dpi=200)
# sns.set_context('talk')
# sns.set_theme(style="whitegrid")
# # Defining a custom style with thick borders
# custom_style = {
#     'axes.spines.left': True,
#     'axes.spines.bottom': True,
#     'axes.spines.right': True,
#     'axes.spines.top': True,
#     'axes.edgecolor': 'black',
#     'axes.linewidth': 1.5,
#     'grid.linestyle': '--',
#     'grid.alpha': 0.7
# }
# # Updating matplotlib's rc settings with the custom style
# plt.rcParams.update(custom_style)
# # set y-axis range to 0-1
# # plt.ylim(0, 1.1)
# sns.lineplot(data=df, x='iter', y='loss')
# plt.tight_layout()
# plt.savefig(f'{file}.png')
# plt.show()