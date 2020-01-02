""" Module containing the plot utils. Each function returns a matplotlib object
    """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.index import Index
import shap
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)


def plot_output_train_test(
        model, data, bins=80, raw=True, labels=None, **kwds):
    """
    Plot the BDT output for the signal and background distributions
    both for training and test set.

    Input
    ----------------------------------------
    model: xgboost or sklearn model

    data: list
    Contains respectively: training
    set dataframe, training label array,
    test set dataframe, test label array

    bins: int or sequence of scalars or str
    If bins is an int, it defines the number of equal-width
    bins in the given range (10, by default). If bins is a
    sequence, it defines a monotonically increasing array of
    bin edges, including the rightmost edge, allowing for
    non-uniform bin widths.
    If bins is a string, it defines the method used to
    calculate the optimal bin width, as defined by
    np.histogram_bin_edges:
    https://docs.scipy.org/doc/numpy/reference/generated
    /numpy.histogram_bin_edges.html#numpy.histogram_bin_edges

    raw: Bool
    If true enables the raw untransformed margin value

    labels: list
    Contains the labels to be displayed in the legend
    If None the labels are class1, class2, ..., classN

    **kwds: extra arguments are passed on to plt.hist()

    Output
    ----------------------------------------
    matplotlib object with the BDT output


    """
    class_labels = np.unique(data[1])
    n_classes = len(class_labels)
    if not raw and n_classes > 2:
        print('Warning: only raw or probability predictions supported with multi-classification')
        raw = True

    if labels is None:
        if n_classes > 2:
            labels = ['class{}'.format(i_class) for i_class, _ in enumerate(class_labels)]
        else:
            labels = ['Signal', 'Background']

    prediction = []
    for xxx, yyy in ((data[0], data[1]), (data[2], data[3])):
        for class_lab in class_labels:
            prediction.append(model.predict(xxx[yyy == class_lab], output_margin=raw))

    low = min(np.min(d) for d in prediction)
    high = max(np.max(d) for d in prediction)
    low_high = (low, high)

    res = []
    # only one figure in case of binary classification
    if n_classes <= 2:
        res.append(plt.figure())
        plt.hist(prediction[1], color='b', alpha=0.5, range=low_high, bins=bins,
                 histtype='stepfilled', label='{} pdf Training Set'.format(labels[1]), **kwds)
        plt.hist(prediction[0], color='r', alpha=0.5, range=low_high, bins=bins,
                 histtype='stepfilled', label='{} pdf Training Set'.format(labels[0]), **kwds)

        hist, bins = np.histogram(
            prediction[3], bins=bins, range=low_high, **kwds)
        scale = len(prediction[1]) / sum(hist)
        err = np.sqrt(hist) * scale
        center = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(center, hist * scale, yerr=err, fmt='o',
                     c='b', label='{} pdf Test Set'.format(labels[1]))

        hist, bins = np.histogram(
            prediction[2], bins=bins, range=low_high, **kwds)
        scale = len(prediction[0]) / sum(hist)
        err = np.sqrt(hist) * scale
        plt.errorbar(center, hist * scale, yerr=err, fmt='o',
                     c='r', label='{} pdf Test Set'.format(labels[0]))

        plt.xlabel('BDT output', fontsize=13, ha='right', position=(1, 20))
        plt.ylabel(r'                                Counts (arb. units)', fontsize=13)
        plt.legend(frameon=False, fontsize=12)
    # n figures in case of multi-classification with n classes
    else:
        cmap = plt.cm.get_cmap('tab10')
        for i_output, _ in enumerate(class_labels):
            res.append(plt.figure())
            for i_class, lab in enumerate(labels):
                plt.hist(prediction[i_class][:, i_output], alpha=0.5, range=low_high, bins=bins,
                         color=cmap(i_class), histtype='stepfilled',
                         label='{} pdf Training Set'.format(lab), **kwds)

                hist, bins = np.histogram(
                    prediction[i_class+n_classes][:, i_output], bins=bins, range=low_high, **kwds)
                scale = len(prediction[i_class][:, i_output]) / sum(hist)
                err = np.sqrt(hist) * scale
                center = (bins[:-1] + bins[1:]) / 2

                plt.errorbar(center, hist * scale, yerr=err, fmt='o',
                             c=cmap(i_class), label='{} pdf Test Set'.format(lab))

            plt.xlabel('BDT output for {}'.format(labels[i_output]), fontsize=13, ha='right',
                       position=(1, 20))
            plt.ylabel(r'                                Counts (arb. units)', fontsize=13)
            plt.legend(frameon=False, fontsize=12)

    return res


def plot_distr(list_of_df, column=None, figsize=None, bins=50, log=False, labels=None):
    """
    Build a DataFrame and create a dataset for each class

    Draw histogram of the DataFrame's series comparing the
    distribution of each class.

    Input
    -----------------------------------------
    list_of_df: list
    Contains a dataframe for each class

    column: list
    Contains the name of the features you want to plot
    Example: ['dEdx', 'pT', 'ct']

    figsize: list
    The size in inches of the figure to create. Uses the value in matplotlib.rcParams by default.

    bins: int or sequence of scalars or str
    If bins is an int, it defines the number of equal-width
    bins in the given range (10, by default). If bins is a
    sequence, it defines a monotonically increasing array of
    bin edges, including the rightmost edge, allowing for
    non-uniform bin widths.

    log: Bool
    If True enable log scale plot

    labels: list
    Contains the labels to be displayed in the legend
    If None the labels are class1, class2, ..., classN

    Output
    -----------------------------------------
    matplotlib object with the distributions of the features for
    signal and background

    """

    if column is not None:
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]
        for dfm in list_of_df:
            dfm = dfm[column]

    if figsize is None:
        figsize = [20, 15]

    if labels is None:
        labels = ['class{}'.format(i_class) for i_class, _ in enumerate(list_of_df)]

    res = plt.figure()
    for i_class, (dfm, lab) in enumerate(zip(list_of_df, labels)):
        if i_class == 0:
            axes = dfm.hist(column=column, alpha=0.5, bins=bins, figsize=figsize, label=lab,
                            density=True, grid=False, log=log)
            axes = axes.flatten()
            axes = axes[:len(column)]
        else:
            dfm.hist(ax=axes, column=column, alpha=0.5, bins=bins, figsize=figsize, label=lab,
                     density=True, grid=False, log=log)
    for axs in axes:
        axs.set_ylabel('Counts (arb. units)')
    plt.legend(loc='best')
    return res


def plot_corr(sig_df, bkg_df, columns, **kwds):
    """
    Calculate pairwise correlation between features for
    two classes (ex: signal and background)

    Input
    -----------------------------------------------
    sig_df: signal candidates dataframe
    bkg_df: background candidates dataframe

    columns: list
    Contains the name of the features you want to plot
    Example: ['dEdx', 'pT', 'ct']

    **kwds: extra arguments are passed on to DataFrame.corr()

    Output
    ------------------------------------------------
    matplotlib object with the correlations between the
    features for signal and background

    """

    data_sig = sig_df[columns]
    data_bkg = bkg_df[columns]

    corrmat_sig = data_sig.corr(**kwds)
    corrmat_bkg = data_bkg.corr(**kwds)

    fig = plt.figure(figsize=(20, 10))
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
    """
    Plot the BDT efficiency calculated with the function
    bdt_efficiency_array() in analysis_utils

    Input
    -----------------------------------
    threshold: array
    Score threshold array

    eff_sig: array
    bdt efficiency array

    Output
    -----------------------------------
    matplotlib object of the bdt efficiency as a
    function of the threshold score

    """
    res = plt.figure()
    plt.plot(threshold, eff_sig, 'r.', label='Signal efficiency')
    plt.legend()
    plt.xlabel('BDT Score')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs Score')
    plt.grid()
    return res


def plot_roc(y_truth, y_score, pos_label=None):
    """
    Calculate and plot the roc curve

    Input
    -------------------------------------

    y_truth : array
    True binary labels. If labels are not either
    {-1, 1} or {0, 1}, then pos_label should be
    explicitly given.

    y_score : array
    Target scores, can either be probability estimates
    of the positive class, confidence values, or
    non-thresholded measure of decisions (as returned
    by “decision_function” on some classifiers).

    pos_label : int or str
    The label of the positive class. When pos_label=None,
    if y_true is in {-1, 1} or {0, 1}, pos_label is set to 1,
    otherwise an error will be raised.



    Output
    -------------------------------------
    matplotlib object with the roc curve

    """

    fpr, tpr, _ = roc_curve(y_truth, y_score, pos_label=pos_label)
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
    return res


def plot_feature_imp(df_in, y_truth, model, n_sample=10000):
    """
    Calculate the feature importance using the shap violin plot for
    each feature. The calculation is performed on a subsample of the
    input training/test set

    Input
    -------------------------------------------
    df_in: Pandas dataframe
    Training or test set dataframe

    y_truth: array
    Training or test set label
    model: trained model

    n_sample: int
    Number of candidates employed to fill
    the shap violin plots.

    Output
    -------------------------------------------
    matplotlib object with shap feature importance

    """
    subs_bkg = df_in[y_truth == 0].sample(n_sample)
    subs_sig = df_in[y_truth == 1].sample(n_sample)
    df_subs = pd.concat([subs_bkg, subs_sig]).sample(frac=1.)
    explainer = shap.TreeExplainer(model.get_original_model())
    shap_values = explainer.shap_values(df_subs)
    res = plt.figure()
    shap.summary_plot(shap_values, df_subs, show=False)
    return res


def plot_precision_recall(y_truth, y_score, pos_label=None):
    """ Plot precision recall curve

    Input
    -------------------------------------
    y_truth: array
    True binary labels. If labels are not either
    {-1, 1} or {0, 1}, then pos_label should be
    explicitly given.

    y_score: array
    Estimated probabilities or decision function.

    pos_label : int or str
    The label of the positive class. When pos_label=None,
    if y_true is in {-1, 1} or {0, 1}, pos_label is set to 1,
    otherwise an error will be raised.


    Output
    -------------------------------------
    matplotlib object with the precision
    recall curve

    """
    precision, recall, _ = precision_recall_curve(
        y_truth, y_score, pos_label=pos_label)
    res = plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    average_precision = average_precision_score(y_truth, y_score)
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))

    return res
