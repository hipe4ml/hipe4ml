"""
Module containing the analysis utils.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.preprocessing import label_binarize


def bdt_efficiency_array(y_truth, y_score, n_points=50, keep_lower=False):
    """
    Calculate the model efficiency as a function of the score
    threshold. The candidates for each class should be labeled
    with 0, ..., N. In case of binary classification, 0 should
    correspond to the background while 1 to the signal

    Input
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

    Output
    ------------------------------------------------
    efficiency: numpy array
        Efficiency as a function of the threshold value
        Numpy array of numpy arrays in case of multi-classification

    threshold: numpy array
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

    Input
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

    Output
    -----------------------------------------------
    score_array: numpy array
        Score array corresponding to efficiency_selected
    """
    score_list = []
    eff, score = bdt_efficiency_array(y_truth, y_score, n_points=1000, keep_lower=keep_lower)
    for eff_val in efficiency_selected:
        interp = InterpolatedUnivariateSpline(score, eff-eff_val)
        score_list.append(interp.roots()[0])
    score_array = np.array(score_list)
    return score_array
