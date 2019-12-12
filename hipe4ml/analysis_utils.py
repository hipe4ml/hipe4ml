""" Module containing the analysis utils.
    """

import numpy as np


def bdt_efficiency_array(y_truth, y_score, n_points=50):
    """
    Calculate the BDT efficiency as a function of the score
    threshold. The signal and the background candidates
    should be labeled respectively with 1 and 0

    Input
    ------------------------------------------------
    y_truth: array
    Training or test set labels. Background candidates should
    be labeled with 0, signal candidates with 1

    y_score: array
    Estimated probabilities or decision function.

    n_points: int
    Number of points to be sampled

    Output
    ------------------------------------------------
    efficiency: numpy array
    Efficiency array as a function of the threshold value

    threshold: numpy array
    Threshold values array


    """
    min_score = np.min(y_score)
    max_score = np.max(y_score)

    threshold = np.linspace(min_score, max_score, n_points)

    n_sig = np.sum(y_truth)

    efficiency = np.empty((0, n_points))

    for thr in threshold:
        n_sig_selected = np.sum(y_truth[y_score > thr])
        efficiency = np.append(efficiency, [n_sig_selected/n_sig])

    return efficiency, threshold
