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
from sklearn.preprocessing import label_binarize


def plot_output_train_test(
        model, data, bins=80, raw=True, labels=None, **kwds):
    """
    Plot the BDT output distributions for each class and output
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
    list of matplotlib objects with the BDT output
    distributions for each class


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
    array of matplotlib axes with the distributions
    of the features for each class

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
    return axes


def plot_corr(list_of_df, columns, labels=None, **kwds):
    """
    Calculate pairwise correlation between features for
    each class (e.g. signal and background in case of binary
    classification)

    Input
    -----------------------------------------------
    list_of_df: list
    Contains dataframes for each class

    columns: list
    Contains the name of the features you want to plot
    Example: ['dEdx', 'pT', 'ct']

    labels: list
    Contains the labels to be displayed in the legend
    If None the labels are class1, class2, ..., classN

    **kwds: extra arguments are passed on to DataFrame.corr()

    Output
    ------------------------------------------------
    list of matplotlib objects with the correlations
    between the features for each class

    """

    corr_mat = []
    for dfm in list_of_df:
        dfm = dfm[columns]
        corr_mat.append(dfm.corr(**kwds))

    if labels is None:
        labels = []
        if len(corr_mat) > 2:
            for i_mat, _ in enumerate(corr_mat):
                labels.append('class{}'.format(i_mat))
        else:
            labels.append('Signal')
            labels.append('Background')

    fig = []
    for mat, lab in zip(corr_mat, labels):
        fig.append(plt.figure(figsize=(8, 7)))
        grid = ImageGrid(fig[-1], 111, axes_pad=0.15, nrows_ncols=(1, 1), share_all=True,
                         cbar_location='right', cbar_mode='single', cbar_size='7%', cbar_pad=0.15)

        opts = {'cmap': plt.get_cmap(
            'coolwarm'), 'vmin': -1, 'vmax': +1, 'snap': True}

        axs = grid[0]
        heatmap = axs.pcolor(mat, **opts)
        axs.set_title(lab, fontsize=14, fontweight='bold')

        lab = mat.columns.values

        # shift location of ticks to center of the bins
        axs.set_xticks(np.arange(len(lab)), minor=False)
        axs.set_yticks(np.arange(len(lab)), minor=False)
        axs.set_xticklabels(lab, minor=False, ha='left', rotation=90, fontsize=10)
        axs.set_yticklabels(lab, minor=False, va='bottom', fontsize=10)
        axs.tick_params(axis='both', which='both', direction="in")

        for tick in axs.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')

        axs.cax.colorbar(heatmap)

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


def plot_roc(y_truth, y_score, labels=None, pos_label=None):
    """
    Calculate and plot the roc curve

    Input
    -------------------------------------

    y_truth : array
    True labels for the belonging class. If labels are not
    {0, 1, ..., N}, then pos_label should be explicitly given.

    y_score : array
    Target scores, can either be probability estimates
    of the positive class, confidence values, or
    non-thresholded measure of decisions (as returned
    by “decision_function” on some classifiers).

    labels: list
    Contains the labels to be displayed in the legend
    If None the labels are class1, class2, ..., classN

    pos_label : int or str
    The label of the positive class. When pos_label=None,
    if y_true is in {0, 1, ..., N}, pos_label is set to 1,
    otherwise an error will be raised.



    Output
    -------------------------------------
    matplotlib object with the roc curves

    """
    # get number of classes
    n_classes = len(np.unique(y_truth))

    if (labels is None and n_classes > 2) or (labels and len(labels) != n_classes):
        labels = []
        for i_class in range(n_classes):
            labels.append('class{}'.format(i_class))

    res = plt.figure()
    if n_classes <= 2:
        fpr, tpr, _ = roc_curve(y_truth, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)' % (roc_auc))
    else:
        cmap = plt.cm.get_cmap('tab10')
        fpr, tpr, roc_auc = (dict() for i_dict in range(3))
        # convert multi-class labels to multi-labels to obtain roc curves
        y_truth_multi = label_binarize(y_truth, classes=range(n_classes))
        for clas, lab in enumerate(labels):
            fpr[clas], tpr[clas], _ = roc_curve(y_truth_multi[:, clas], y_score[:, clas])
            roc_auc[clas] = auc(fpr[clas], tpr[clas])
            plt.plot(fpr[clas], tpr[clas], lw=1, c=cmap(clas),
                     label='{0} (AUC = {1:.4f})'.format(lab, roc_auc[clas]))
        # compute also micro average
        fpr['micro'], tpr['micro'], _ = roc_curve(y_truth_multi.ravel(), y_score.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        plt.plot(fpr['micro'], tpr['micro'], lw=1, linestyle='--', c='black',
                 label='average (AUC = {:.4f})'.format(roc_auc['micro']))

    plt.plot([0, 1], [0, 1], '-.', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right')
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
    If larger than the number of candidates
    in each class, minimum number of candidates
    in a given class used instead

    Output
    -------------------------------------------
    list of matplotlib objects with shap feature importance

    """
    class_labels, class_counts = np.unique(y_truth, return_counts=True)
    n_classes = len(class_labels)
    for class_count in class_counts:
        if n_sample > class_count:
            n_sample = class_count

    subs = []
    for i_class, class_lab in enumerate(class_labels):
        subs.append(df_in[y_truth == class_lab].sample(n_sample))

    df_subs = pd.concat(subs).sample(frac=1.)
    explainer = shap.TreeExplainer(model.get_original_model())
    shap_values = explainer.shap_values(df_subs, approximate=True)

    res = []
    if n_classes <= 2:
        res.append(plt.figure(figsize=(18, 9)))
        shap.summary_plot(shap_values, df_subs, plot_size=(18, 9), show=False)
    else:
        for i_class in range(n_classes):
            res.append(plt.figure(figsize=(18, 9)))
            shap.summary_plot(shap_values[i_class], df_subs, plot_size=(18, 9), show=False)
        res.append(plt.figure(figsize=(18, 9)))
        shap.summary_plot(shap_values, df_subs, plot_type='bar', plot_size=(18, 9), show=False)

    return res


def plot_precision_recall(y_truth, y_score, labels=None, pos_label=None):
    """ Plot precision recall curve

    Input
    -------------------------------------
    y_truth: array
    True labels for the belonging class. If labels are not
    {0, 1, ..., N}, then pos_label should be explicitly given.

    y_score: array
    Estimated probabilities or decision function.

    pos_label : int or str
    The label of the positive class. When pos_label=None,
    if y_true is in {0, 1, ..., N}, pos_label is set to 1,
    otherwise an error will be raised.


    Output
    -------------------------------------
    matplotlib object with the precision
    recall curves

    """
    # get number of classes
    n_classes = len(np.unique(y_truth))

    if (labels is None and n_classes > 2) or (labels and len(labels) != n_classes):
        labels = []
        for i_class in range(n_classes):
            labels.append('class{}'.format(i_class))

    res = plt.figure()
    if n_classes <= 2:
        precision, recall, _ = precision_recall_curve(y_truth, y_score, pos_label=pos_label)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
        average_precision = average_precision_score(y_truth, y_score)
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    else:
        cmap = plt.cm.get_cmap('tab10')
        precision, recall = (dict() for i_dict in range(2))
        # convert multi-class labels to multi-labels to obtain a curve for each class
        y_truth_multi = label_binarize(y_truth, classes=range(n_classes))
        for clas, lab in enumerate(labels):
            precision[clas], recall[clas], _ = precision_recall_curve(
                y_truth_multi[:, clas], y_score[:, clas], pos_label=pos_label)
            plt.step(recall[clas], precision[clas], color=cmap(clas), lw=1, where='post',
                     label=lab)
        # compute also micro average
        precision['micro'], recall['micro'], _ = precision_recall_curve(
            y_truth_multi.ravel(), y_score.ravel())
        plt.step(recall['micro'], precision['micro'], color='black', where='post',
                 linestyle='--', lw=1, label='average')
        average_precision = average_precision_score(y_truth_multi, y_score, average='micro')
        plt.title('Average precision score, micro-averaged over all classes: {0:0.2f}'
                  .format(average_precision))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='lower left')
    if n_classes > 2:
        plt.grid()
    return res
