""" Module containing the plot utils. Each function returns a matplotlib object
    """
from inspect import signature
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from pandas.core.index import Index
import pandas as pd
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)

import shap


# plot the BDT score distribution in the train and in the test set for both signal and background
def plot_output_train_test(
        clf, x_train, y_train, x_test, y_test, model='xgb', features=None, bins=80):
    '''
    Plot the BDT output for the signal and bckground distributions
    both for training and test set.

    Input
    ----------------------------------------
    x_train: training set
    y_train: training label
    x_test: test set
    y_test: test label
    model: Could be 'xgb' or 'sklearn'
    features: training columns

    Output
    ----------------------------------------
    res: matplotlib object with the BDT output


    '''

    prediction = []
    for xxx, yyy in ((x_train, y_train), (x_test, y_test)):
        if model == 'xgb':
            df1 = clf.predict(xxx[yyy > 0.5][features], output_margin=True)
            df2 = clf.predict(xxx[yyy < 0.5][features], output_margin=True)
        elif model == 'sklearn':
            df1 = clf.decision_function(xxx[yyy > 0.5]).ravel()
            df2 = clf.decision_function(xxx[yyy < 0.5]).ravel()
        else:
            print('Error: wrong model type used')
            return None
        prediction += [df1, df2]

    low = min(np.min(d) for d in prediction)
    high = max(np.max(d) for d in prediction)
    low_high = (low, high)

    res = plt.figure()

    plt.hist(prediction[1], color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True, log=True, label='Background pdf Training Set')
    plt.hist(prediction[0], color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True, log=True, label='Signal pdf Training Set')

    hist, bins = np.histogram(
        prediction[2], bins=bins, range=low_high, density=True)
    scale = len(prediction[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o',
                 c='r', label='Background pdf Test Set')

    hist, bins = np.histogram(
        prediction[3], bins=bins, range=low_high, density=True)
    scale = len(prediction[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o',
                 c='b', label='Signal pdf Test Set')

    # plt.gcf().subplots_adjust(left=0.14)
    plt.xlabel('BDT output', fontsize=13, ha='right', position=(1, 20))
    plt.ylabel(r'                                Counts (arb. units)', fontsize=13)
    plt.legend(frameon=False, fontsize=12)
    return res


def plot_distr(sig_df, bkg_df, column=None, figsize=None, bins=50, log=False):
    """Build a DataFrame and create two dataset for signal and bkg

    Draw histogram of the DataFrame's series comparing the distribution
    in `data1` to `data2`.

    Input
    -----------------------------------------
    sig_df: signal candidates dataframe
    bkg_df: background candidates dataframe
    column: training columns
    log: log scale plot

    Output
    -----------------------------------------
    res: matplotlib object with the distributions of the features for
    signal and background

    """

    data1 = bkg_df
    data2 = sig_df

    if column is not None:
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]
        data1 = data1[column]
        data2 = data2[column]

    if figsize is None:
        figsize = [20, 15]

    res = plt.figure()
    axes = data1.hist(column=column, color='tab:blue', alpha=0.5, bins=bins, figsize=figsize,
                      label='Background', density=True, grid=False, log=log)
    axes = axes.flatten()
    axes = axes[:len(column)]
    data2.hist(ax=axes, column=column, color='tab:orange', alpha=0.5, bins=bins, label='Signal',
               density=True, grid=False, log=log)[0].legend()
    for axs in axes:
        axs.set_ylabel('Counts (arb. units)')
    return res


def plot_corr(sig_df, bkg_df, columns, **kwds):
    """Calculate pairwise correlation between features.

    Input
    -----------------------------------------------
    sig_df: signal candidates dataframe
    bkg_df: background candidates dataframe
    column: training columns
    **kwds: extra arguments are passed on to DataFrame.corr()

    Output
    ------------------------------------------------
    fig: matplotlib object with the correlations between the
    features for signal and background

    """

    data_sig = sig_df[columns]
    data_bkg = bkg_df[columns]

    corrmat_sig = data_sig.corr(**kwds)
    corrmat_bkg = data_bkg.corr(**kwds)

    tit = r'$\mathrm{\ \ \ ALICE \ Simulation}$ Pb-Pb $\sqrt{s_{\mathrm{NN}}}$ = 5.02 TeV'
    fig = plt.figure(figsize=(20, 10))
    # plt.title(t,y=1.08,fontsize=16)
    plt.suptitle(tit, fontsize=18, ha='center')
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.15, share_all=True,
                     cbar_location='right', cbar_mode='single', cbar_size='7%', cbar_pad=0.15)

    opts = {'cmap': plt.get_cmap(
        'coolwarm'), 'vmin': -1, 'vmax': +1, 'snap': True}

    ax1 = grid[0]
    ax2 = grid[1]
    heatmap1 = ax1.pcolor(corrmat_sig, **opts)
    heatmap2 = ax2.pcolor(corrmat_bkg, **opts)
    ax1.set_title('Signal', fontsize=14, fontweight='bold')
    ax2.set_title('Background', fontsize=14, fontweight='bold')

    lab = corrmat_sig.columns.values
    for axs in (ax1,):
        # shift location of ticks to center of the bins
        axs.set_xticks(np.arange(len(lab)), minor=False)
        axs.set_yticks(np.arange(len(lab)), minor=False)
        axs.set_xticklabels(lab, minor=False, ha='left',
                            rotation=90, fontsize=10)
        axs.set_yticklabels(lab, minor=False, va='bottom', fontsize=10)
        axs.tick_params(axis='both', which='both', direction="in")

        for tick in axs.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')

    for axs in (ax2,):
        # shift location of ticks to center of the bins
        axs.set_xticks(np.arange(len(lab)), minor=False)
        axs.set_yticks(np.arange(len(lab)), minor=False)
        axs.set_xticklabels(lab, minor=False, ha='left',
                            rotation=90, fontsize=10)
        axs.tick_params(axis='both', which='both', direction="in")
        for tick in axs.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')

    ax1.cax.colorbar(heatmap1)
    ax2.cax.colorbar(heatmap2)
    ax1.cax.toggle_label(True)
    return fig


def plot_bdt_eff(threshold, eff_sig):
    """Plot the BDT efficiency calculated with the function bdt_efficiency_array() in training_utils.py

    Input
    -----------------------------------
    threshold: score threscold array
    eff_sig: bdt efficiency array

    Output
    -----------------------------------
    res: matplotlib object of the bdt efficiency as a function of the threshold score

    """
    res = plt.figure()
    plt.plot(threshold, eff_sig, 'r.', label='Signal efficiency')
    plt.legend()
    plt.xlabel('BDT Score')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs Score')
    plt.grid()
    return res


def plot_roc(y_truth, model_decision):
    """Calculate and plot the roc curve

    Input
    -------------------------------------
    y_truth: test set label
    model decision: predicted score

    Output
    -------------------------------------
    res: matplotlib object with the roc curve

    """
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_truth, model_decision)
    roc_auc = auc(fpr, tpr)
    res = plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right")
    plt.grid()
    plt.close()
    return res


def plot_feature_imp(df_in, y_lab, model):
    """Calculate the feature importance using the shap violin plot for each feature. The calculation
    is performed on a subsample of the input training/test set. The model could be sklearn or xgboost

    Input
    -------------------------------------------
    df_in: test set dataframe
    y_lab: test set label
    model: trained model

    Output
    -------------------------------------------
    res: matplotlib object with shap feature importance

    """
    subs_bkg = df_in[y_lab == 0].sample(10000)
    subs_sig = df_in[y_lab == 1].sample(10000)
    df_subs = pd.concat([subs_bkg, subs_sig]).sample(frac=1.)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_subs)
    res = plt.figure()
    shap.summary_plot(shap_values, df_subs, show=False)
    return res


def plot_precision_recall(y_test, y_score):
    """ Plot precision recall curve

    Input
    -------------------------------------
    y_test: test set label
    y_score: predicted score

    Output
    -------------------------------------
    res: matplotlib object with the precision recall curve

    """
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    res = plt.figure()
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    average_precision = average_precision_score(y_test, y_score)
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))

    return res
