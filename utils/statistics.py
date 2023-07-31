import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import ttest_1samp

from keras import backend as K


def chi_goodness_of_fit_test(data, col_data, col_name, significance_level):
    chi_goodness_of_fit_result = stats.chisquare(col_data)

    null_hypothesis = f'There is no significant difference between {col_name} and the expected frequencies'
    # {col_name} is not representitive of the population at alpha={significance_level} when the null hypothesis is rejected.
    chi_square = chi_goodness_of_fit_result.statistic
    p = chi_goodness_of_fit_result.pvalue
    if p <= significance_level:
        reject_null_hypothesis = "Yes"
    else:
        reject_null_hypothesis = "No"

    data = pd.concat([data,
                      pd.DataFrame(
                    {"Variable": col_name,
                    "Chi-square": chi_square,
                    "P-value": p,
                    # "Null Hypothesis": null_hypothesis,
                    f"Reject Null Hypothesis at alpha={significance_level}?": reject_null_hypothesis},
                    index=[0])],
                    ignore_index=True)
    return data


def chi_independence_test(data, col_name, target, chi_square, p, significance_level):
    null_hypothesis = f'{col_name} and {target} are independent of each other'
    if p <= significance_level:
        reject_null_hypothesis = "Yes"
    else:
        reject_null_hypothesis = "No"

    data = pd.concat([data,
                      pd.DataFrame(
                    {"Variable": col_name,
                    "Chi-square": chi_square,
                    "P-value": p,
                    # "Null Hypothesis": null_hypothesis,
                    f"Reject Null Hypothesis at alpha={significance_level}?": reject_null_hypothesis},
                    index=[0])],
                    ignore_index=True)
    return data

def run_chi_tests(data, target, significance_level,
                  plot_title=None, plot_title_y=None,
                  plot_row=None, plot_col=None, figsize=None, plot=True,
                  rotate_x_label_col=[], rotate_angle=None,
                  h_pad=3, print_result_df=True,
                  independence_test=True,
                  goodness_of_fit_test=True):   
    if plot:
        fig, axes = plt.subplots(plot_row, plot_col, figsize=figsize, sharey=True)
        fig.tight_layout(h_pad=h_pad)
        plt.suptitle(plot_title, y=plot_title_y)


    chi_independence_df = pd.DataFrame(columns=[
        "Independent Variable",
        "Chi-square",
        "P-value",
        # "Null Hypothesis",
        f"Reject Null Hypothesis at alpha={significance_level}?"
        ])
    chi_goodness_of_fit_df = pd.DataFrame(columns=[
        "Independent Variable",
        "Chi-square",
        "P-value",
        # "Null Hypothesis",
        f"Reject Null Hypothesis at alpha={significance_level}?"
        ])
    for i, col in enumerate(data.drop(target, axis=1).columns):
        if plot:
            if (plot_row==1) or (plot_col==1):
                ax = axes[i]
            else:
                ax = axes[i//plot_col, i%plot_col]
            sns.lineplot(data, x=col, y=target, ax=ax).invert_yaxis()
            if col in rotate_x_label_col:
                plt.sca(ax)
                plt.xticks(rotation=rotate_angle)

            ax.set_yticks(sorted(list(data[target].unique())))
            ax.set_ylabel(target, rotation=0)
        x = data[col]
        y = data[target]

        contingency_table = pd.crosstab(x, y)
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        if print_result_df:
            print(f'Contingecy table for {col} and {target}:')
            print(contingency_table, "\n")
            print(f'Expected frequencies for {col} and {target}:')
            print(expected)

        if independence_test:
            "Perform Chi-Square test of Independence."
            chi_independence_df = chi_independence_test(chi_independence_df, col, target, chi2, p, significance_level)
            print(f'''
            If the null hypothesis is not rejected at the significance level of {significance_level},
            the Variable and the target (i.e. {target}) are independent of each other.
            ''')
            
        if goodness_of_fit_test:
            "Perform Chi-square test of goodness of fit."
            chi_goodness_of_fit_df = chi_goodness_of_fit_test(chi_goodness_of_fit_df, x, col, significance_level)

    result_dfs = []
    if independence_test:
        result_dfs.append(chi_independence_df.sort_values(["Chi-square", "P-value"],
                                                          ascending=[False, True]).set_index('Variable'))
    if goodness_of_fit_test:
        result_dfs.append(chi_goodness_of_fit_df.sort_values(["Chi-square", "P-value"],
                                                             ascending=[False, True]).set_index('Variable'))
    return result_dfs


def hypothesis_test(values, target, alpha, model, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout(pad=5)
    print(f"Hypothesis test for {type(model).__name__}")
    t_stat, pvalue = ttest_1samp(values, target)
    null_hypothesis = f"Mean test score is not significantly different from {target} at alpha={alpha}."
    print(f"Null hypothesis - {null_hypothesis}\n")

    print("Result:")
    if pvalue/2 > alpha:
        print(f"Failed to reject the null hypothesis: {null_hypothesis}")
    else:
        if t_stat > 0:
            print(f"Rejected the null hypothesis - Mean test score is greater than {target} at alpha={alpha}.")
        else:
            print(f"Rejected the null hypothesis - Mean test score is smaller than {target} at alpha={alpha}.")
    print()

    sns.histplot(values, ax=ax)
    # ax.vlines(np.mean(scores), 0, 10, colors='red')
    ax.text(0, 0,
        f"""
        Mean: {round(np.mean(values), 4)}
        Std: {round(np.std(values), 4)}
        Max: {round(np.max(values), 4)}
        Min: {round(np.min(values), 4)}
        """)
    ax.set_title(type(model).__name__)
    ax.set_xlabel("Accuracy Score")
    ax.set_xbound(0, 1)


def f1_score(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1_score