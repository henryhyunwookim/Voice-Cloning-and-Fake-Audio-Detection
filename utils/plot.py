import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pylab import rcParams

def plot_histograms(data,
                    target, target_figsize,
                    dependent_layout, dependent_figsize,
                    include_boxplots=False):
    print(f"Distribution of target '{target}' and dependent variables:")
    data.hist(layout=dependent_layout, figsize=dependent_figsize, sharey=True, grid=False)
    plt.tight_layout();

    if include_boxplots:
        data.plot(kind='box', subplots=True, layout=dependent_layout, figsize=dependent_figsize)
        plt.tight_layout();


def plot_scores(scores_dict):
    fig, axes = plt.subplots(len(scores_dict.keys()), 1,
                    figsize=(5, len(scores_dict.keys())*3))
    fig.tight_layout(pad=5)
    for i, (function, scores) in enumerate(scores_dict.items()):
        try:
            ax = axes[i]
        except TypeError:
            ax = axes
        ax.text(0.02, 0,
    f"""
    Mean: {round(np.mean(scores), 2)}
    Std: {round(np.std(scores), 2)}
    Max: {round(np.max(scores), 2)}
    75th Percentile: {round(np.percentile(scores, 75), 2)}
    Median: {round(np.median(scores), 2)}
    25th Percentile: {round(np.percentile(scores, 25), 2)}
    Min: {round(np.min(scores), 2)}
    """)
        sns.distplot(scores, ax=ax)
        ax.set_title(function)
        ax.set_xlabel("Accuracy Score")
        ax.set_xbound(0, 1)


def plot_distributions(data, columns, hue_col):
    fig, axes = plt.subplots(len(columns), 2, figsize=(10, len(columns)*2))
    for i, col in enumerate(columns):
        sns.kdeplot(data=data, x=col, hue=hue_col, common_norm=False, ax=axes[i, 0])
        sns.histplot(data=data, x=col, hue=hue_col, common_norm=False, ax=axes[i, 1],
                    stat='density', element='step', fill=False, cumulative=True)
    plt.tight_layout(pad=3)
    plt.suptitle("Kernel Density Estimate (KDE) Plot & Cumulative Distribution Function (CDF) Plot", y=1.01);


def plot_bars(data, ax, stacked=True):
    ax = data.plot(kind='bar', stacked=stacked, rot=0, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    ax.legend(loc='lower center');


def plot_pies(values, labels, ax, title, autopct='%1.1f%%', startangle=90, sort_values=True, sort_reverse=False):
    if sort_values:
        sorted_dict = dict(sorted(zip(labels, values), key=operator.itemgetter(1), reverse=sort_reverse))
        values = sorted_dict.values()
        labels = sorted_dict.keys()

    ax.pie(values,
           labels=labels,
           autopct=autopct,
           startangle=startangle)
    ax.set_title(title);


def plot_timeseries(data, time_variable, group_col, group_vals, figsize,
                    plot_ratio_for_binary_class=None, class_names=None):
    count_df = data.groupby([time_variable, group_col]).count().reset_index().iloc[:,:3]
    last_col = count_df.columns[2]

    if not plot_ratio_for_binary_class:
        fig, ax = plt.subplots(figsize=figsize)
        for val in group_vals:
            count_df[count_df[group_col]==val].set_index(time_variable).rename(columns={last_col:val}).plot(ax=ax)

    else:
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        count_dfs = []
        for val in group_vals:
            to_plot = count_df[count_df[group_col]==val].set_index(time_variable).rename(columns={last_col:val})
            count_dfs.append(to_plot)
            to_plot.plot(ax=ax[0])
    ax[0].set_title(f"Value counts of {group_col} for each {time_variable}")

    ratios = count_dfs[0][group_vals[0]].values / count_dfs[1][group_vals[1]].values
    indices = count_dfs[0].index
    sns.lineplot(
        dict(zip(indices, ratios)),
        ax=ax[1])
    ax[1].set_title(f"{class_names[0]} to {class_names[1]} ratio for each {time_variable}")
    plt.tight_layout(pad=3);


def plot_heatmap(df, figsize, rotate_xticks=None, corr_df=False,
                 vmin=None, vmax=None, center=None):
    plt.figure(figsize=figsize)
    if corr_df:
        sns.heatmap(df, vmin=vmin, vmax=vmax, center=center)
    else:
        sns.heatmap(df.corr(), vmin=vmin, vmax=vmax, center=center)
    plt.tight_layout()

    if rotate_xticks != None:
        plt.xticks(rotation = rotate_xticks)


def plot_autocorrelation(df, column, partial):
    rcParams['figure.figsize'] = 12, 2
    if partial:
        plot_pacf(df[column], method='ywm');
    else:
        plot_acf(df[column]);


def plot_bollinger_band(rolling_mean, bollinger_band_window, bollinger_band_std,
                        upper_band, lower_band, y_test,
                        figsize=(12, 4), additional_df=None, xlabel=None, ylabel=None):
    ax = rolling_mean.plot(label=f'{bollinger_band_window}-day SMA += {bollinger_band_std} STD',
                           figsize=figsize)
    
    ax.fill_between(y_test.index,
                    lower_band,
                    upper_band,
                    color='b', alpha=.2)
    
    if xlabel != None:
        ax.set_xlabel(xlabel)

    if ylabel != None:
        ax.set_ylabel(ylabel)
    
    if isinstance(additional_df, pd.DataFrame):
        additional_df.plot(ax=ax)

    plt.legend()
    plt.tight_layout()
    plt.show()