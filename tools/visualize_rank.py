import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import Transform
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter, MaxNLocator
# sns.set_theme(style="whitegrid", font_scale=1.2, context="talk", palette=sns.color_palette("bright"), color_codes=False)
# matplotlib.rcParams['mathtext.fontset'] = 'cm'


# Example data

# Apply custom square root scale to the y-axis


def get_arr(dt, k='eval_spearmanr'):
    eval_per = []
    for i in range(len(dt)):
        if k in dt[i].keys():
            eval_per.append(dt[i][k])
    return eval_per

methods = {'ds':'Domain-specific-CA', 'ties':'Ties-CA', 'dare':'Dare-CA'}
tasks = [ 'STSB' ,'MRPC', 'RTE']


metric_name = {'rte':'eval_accuracy', 'stsb':'eval_spearmanr', 'mrpc':'eval_f1', 'sst2': 'eval_accuracy','qnli':'eval_accuracy'}
rank = [1, 2, 4, 6]
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b']

fig, (ax1, ax2) = plt.subplots(2, 3, sharex=True,figsize=(15, 5), gridspec_kw={'height_ratios': [1, 1]})
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for t, color in zip(tasks, colors):
    for i in range(3):
        acc_stsb = []
        for a in rank:
            dir = f'checkpoints/rank_{a}_{list(methods.keys())[i]}/t5-small/{t.lower()}/curve_MEO/7_7_token_2e-4/trainer_state.json'
            with open(dir, 'rb') as f:
                data = json.load(f)
                data = data['log_history']
                accs = get_arr(data, k=metric_name[t.lower()])
                max_acc = max(accs)
                acc_stsb.append(max_acc)
        if t.lower() != 'rte':
            ax1[i].grid(True, linestyle='--')
            ax1[i].plot(rank, acc_stsb, label=t, marker='*', color=color)
            ax1[i].tick_params(axis='x', which='both', bottom=False, top=False)
            ax1[i].tick_params(axis='y', which='major', labelsize=14)
            ax1[i].set_title(methods[list(methods.keys())[i]], fontsize=25)
            ax1[i].set_ylim(0.86,0.92)
            ax1[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        else:
            ax2[i].grid(True, linestyle='--')
            ax2[i].plot(rank, acc_stsb, label=t, marker='*', color=color)
            ax2[i].set_ylim(0.5, 0.75) 
            ax2[i].tick_params(axis='y', which='major', labelsize=14)   
            ax2[i].tick_params(axis='x', which='major', labelsize=14)
            ax2[i].set_xlabel('Rank', fontsize=20)
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
        # Optionally limit the number of ticks

  # Bottom-right diagonal

handles, labels = [], []
for ax in [ax1[0], ax2[0]]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
fig.text(-0.01, 0.6, 'Eval Metric', va='center', rotation='vertical', fontsize=14)
fig.legend(handles, labels,loc='lower center', bbox_to_anchor=(0.5, -0.01),ncol=len(tasks), frameon=True, fontsize=20)
plt.tight_layout(rect=[0, 0.1, 1, 1])
# plt.show()
plt.savefig(f'plot_rank.pdf',format='pdf',transparent=True,bbox_inches="tight")