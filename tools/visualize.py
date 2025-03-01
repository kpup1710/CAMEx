import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter, MaxNLocator
# sns.set_theme(style="whitegrid", font_scale=1.2, context="talk", palette=sns.color_palette("bright"), color_codes=False)
# matplotlib.rcParams['mathtext.fontset'] = 'cm'

def get_arr(dt, k='eval_spearmanr'):
    eval_per = []
    for i in range(len(dt)):
        if k in dt[i].keys():
            eval_per.append(dt[i][k])
    return eval_per

methods = {'ds':'Domain-specific-CA', 'ties':'Ties-CA', 'dare':'Dare-CA'}
tasks = ['rte', 'stsb' ,'mrpc']

# atone is results of model at alpha=1, read more in sec 3 in the paper
atone = {
  'rte': [0.7472, 0.7581, 0.787],
  'stsb': [0.8947, 0.8954, 0.8956],
  'mrpc': [0.9116, 0.9249, 0.9115]
} 
metric_name = {'rte':'eval_accuracy', 'stsb':'eval_spearmanr', 'mrpc':'eval_f1'}
alpha = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b']

fig, (ax1,ax2) = plt.subplots(2, 3, figsize=(20, 5), sharex=True, sharey=False)
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for t, color in zip(tasks, colors):
    for i in range(3):
        acc_stsb = []
        for a in alpha:
            if a != 1.0:
                dir = f'checkpoints/alpha_{a}_{list(methods.keys())[i]}/t5-base/{t}/curve_MEO/7_7_token_2e-4/trainer_state.json'
                with open(dir, 'rb') as f:
                    data = json.load(f)
                    data = data['log_history']
                    accs = get_arr(data, k=metric_name[t])
                    max_acc = max(accs)
                    acc_stsb.append(max_acc)
        acc_stsb.append(atone[t][i])

        if t != 'rte':
            ax1[i].grid(True, linestyle='--')
            ax1[i].plot(alpha, acc_stsb, label=t, marker='*', color=color)
            ax1[i].tick_params(axis='x', which='both', bottom=False, top=False)
            ax1[i].set_ylim(0.86,0.95)
            ax1[i].set_title(methods[list(methods.keys())[i]], fontsize=25)
            ax1[i].tick_params(axis='y', which='major', labelsize=14)
            ax1[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        else:
            ax2[i].grid(True, linestyle='--')
            ax2[i].plot(alpha, acc_stsb, label=t.upper(), marker='*', color=color)
            ax2[i].set_ylim(0.65, 0.8) 
            ax2[i].tick_params(labeltop=False,axis='x', which='major', labelsize=14)
            ax2[i].tick_params(axis='y', which='major', labelsize=14)
            ax2[i].set_xlabel('alpha', fontsize=20)
            ax2[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax1[i].spines['bottom'].set_visible(False)
        ax2[i].spines['top'].set_visible(False)
        d = .01  # Size of diagonal lines
        kwargs = dict(transform=ax1[i].transAxes, color='k', clip_on=False)
        ax1[i].plot((-d, +d), (-d, +d), **kwargs)  # Top-left diagonal
        ax1[i].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

        kwargs.update(transform=ax2[i].transAxes)  # Switch to the bottom subplot's coordinates
        ax2[i].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
        ax2[i].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

        # ax.set_ylabel('eval metric', fontsize=14)
handles, labels = [], []
for ax in [ax1[0], ax2[0]]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l.upper())
fig.text(-0.01, 0.6, 'Eval Metric', va='center', rotation='vertical', fontsize=25)
fig.legend(handles, labels,loc='lower center', bbox_to_anchor=(0.5, -0.04),ncol=len(methods), frameon=True, fontsize=25)
plt.tight_layout(rect=[0, 0.1, 1, 1])
# plt.show()
plt.savefig(f'plot_alpha.pdf',format='pdf',transparent=True,bbox_inches="tight")