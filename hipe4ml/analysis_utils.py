""" Module containing the analysis utils.
    """

import numpy as np
from sklearn.preprocessing import label_binarize


def bdt_efficiency_array(y_truth, y_score, n_points=50):
    """
    Calculate the BDT efficiency as a function of the score
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

    Output
    ------------------------------------------------
    efficiency: numpy array
    Efficiency array as a function of the threshold value
    Numpy array of numpy arrays in case of multi-classification

    threshold: numpy array
    Threshold values array


    """
    # get number of classes
    n_classes = len(np.unique(y_truth))

    min_score = np.min(y_score)
    max_score = np.max(y_score)

    threshold = np.linspace(min_score, max_score, n_points)

    if n_classes <= 2:
        n_sig = np.sum(y_truth)

        efficiency = np.empty((0, n_points))
        for thr in threshold:
            n_sig_selected = np.sum(y_truth[y_score > thr])
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
                n_sig_selected = np.sum(y_truth_multi[:, clas][y_score[:, clas] > thr])
                efficiency = np.append(efficiency, [n_sig_selected/n_sig])
            efficiencies.append(efficiency)
        efficiencies = np.array(efficiencies)

    return efficiencies, threshold
