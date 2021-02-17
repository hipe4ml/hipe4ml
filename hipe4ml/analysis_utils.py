"""
Module containing the analysis utils.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


def bdt_efficiency_array(y_truth, y_score, n_points=50, keep_lower=False):
    """
    Calculate the model efficiency as a function of the score
    threshold. The candidates for each class should be labeled
    with 0, ..., N. In case of binary classification, 0 should
    correspond to the background while 1 to the signal

    Parameters
    ------------------------------------------------
    y_truth: array
        Training or test set labels. The candidates for each
        class should be labeled with 0, ..., N.
        In case of binary classification, 0 should
        correspond to the background while 1 to the signal

    y_score: array
        Estimated probabilities or decision function.

    n_points: int
        Number of points to be sampled

    keep_lower: bool
        If True compute the efficiency using the candidates with
        score lower than the score threshold; otherwise using the
        candidates with score higher than the score threshold.

    Returns
    ------------------------------------------------
    out: numpy array
        Efficiency as a function of the threshold value
        Numpy array of numpy arrays in case of multi-classification

    out: numpy array
        Threshold values
    """
    operator = np.greater
    if keep_lower:
        operator = np.less
    # get number of classes
    n_classes = len(np.unique(y_truth))

    min_score = np.min(y_score)
    max_score = np.max(y_score)

    threshold = np.linspace(min_score, max_score, n_points)

    if n_classes <= 2:
        n_sig = np.sum(y_truth)

        efficiency = np.empty((0, n_points))
        for thr in threshold:
            n_sig_selected = np.sum(y_truth[operator(y_score, thr)])
            efficiency = np.append(efficiency, [n_sig_selected/n_sig])
        efficiencies = efficiency
    else:
        efficiencies = []
        for clas in range(n_classes):
            y_truth_multi = label_binarize(y_truth, classes=range(n_classes))
            # considering signal only the class for the same BDT output
            n_sig = np.sum(y_truth_multi[:, clas])

            efficiency = np.empty((0, n_points))
            for thr in threshold:
                n_sig_selected = np.sum(
                    y_truth_multi[:, clas][operator(y_score[:, clas], thr)])
                efficiency = np.append(efficiency, [n_sig_selected/n_sig])
            efficiencies.append(efficiency)
        efficiencies = np.array(efficiencies)

    return efficiencies, threshold


def score_from_efficiency_array(y_truth, y_score, efficiency_selected, keep_lower=False):
    """
    Return the score array corresponding to an external fixed efficiency
    array.

    Parameters
    -----------------------------------------------
    y_truth: array
        Training or test set labels. The candidates for each
        class should be labeled with 0, ..., N.
        In case of binary classification, 0 should
        correspond to the background while 1 to the signal

    y_score: array
        Estimated probabilities or decision function.

    keep_lower: bool
        If True compute the efficiency using the candidates with
        score lower than the score threshold; otherwise using the
        candidates with score higher than the score threshold.

    efficiency_selected: list or array
        Efficiency array along which calculate
        the corresponding score array

    Returns
    -----------------------------------------------
    out: numpy array
        Score array corresponding to efficiency_selected
    """
    score_list = []
    eff, score = bdt_efficiency_array(
        y_truth, y_score, n_points=1000, keep_lower=keep_lower)
    for eff_val in efficiency_selected:
        interp = InterpolatedUnivariateSpline(score, eff-eff_val)
        score_list.append(interp.roots()[0])
    score_array = np.array(score_list)
    return score_array


def train_test_generator(data_list, labels_list, sliced_df=False, **kwds):
    """
    Return a list containing respectively training set dataframe,
    training label array, test set dataframe, test label array
    computed from a list of TreeHandler objects. If sliced_df == True,
    the method preforms the train-test split for each slice

    Parameters
    -----------------------------------------------
    data_list: list
        List of TreeHandler models. For example: if you perform binary
        classification the list should contain the TreeHandlers corresponding
        to the signal and the background candidates

    labels_list: list
        List containing the labels associated to each DataHandler. For example:
        if you perform binary classification, the list should be [1,0]

    sliced_df: bool
        If True, the function searches for the slices stored in the DataHandler
        and perform the train_test_split for each slice

    **kwds
        Extra arguments are passed on to sklearn.model_selection.train_test_split:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    Returns
    -----------------------------------------------
    out: list
        List containing respectively training set dataframe,
        training label array, test set dataframe, test label array. If sliced_df==True
        returns a list containing the above cited list for each slice
    """
    if sliced_df is False:
        labels_train_test = []
        df_list = []
        for data, labels in zip(data_list, labels_list):
            data_frame = data.get_data_frame()
            labels_train_test += len(data_frame)*[labels]
            df_list.append(data_frame)
        del data_list, data_frame
        df_tot_train_test = pd.concat(df_list, sort=True)
        del df_list
        train_test = train_test_split(df_tot_train_test, np.array(labels_train_test), **kwds)
        # swap for ModelHandler compatibility
        train_test[1], train_test[2] = train_test[2], train_test[1]
        return train_test

    train_test_slices = []
    n_slices = len(data_list[0].get_projection_binning())
    for slice_ind in range(n_slices):
        labels_train_test = []
        df_list = []
        for data, labels in zip(data_list, labels_list):
            data_frame = data.get_slice(slice_ind)
            labels_train_test += len(data_frame)*[labels]
            df_list.append(data_frame)
        del data_frame
        df_tot_train_test = pd.concat(df_list, sort=True)
        del df_list
        train_test = train_test_split(df_tot_train_test, np.array(labels_train_test), **kwds)
        train_test[1], train_test[2] = train_test[2], train_test[1]
        train_test_slices.append(train_test)
    return train_test_slices
