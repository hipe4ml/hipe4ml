""" Module containing the analysis utils.
    """

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score


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
                n_sig_selected = np.sum(
                    y_truth_multi[:, clas][y_score[:, clas] > thr])
                efficiency = np.append(efficiency, [n_sig_selected/n_sig])
            efficiencies.append(efficiency)
        efficiencies = np.array(efficiencies)

    return efficiencies, threshold


def cross_val_roc_score_multiclass(model, training_df, y_training, n_classes, n_fold):
    """
    Evaluate a score using the roc auc metric by cross validation for multiclass
    classifiers

    Input
    ------------------------------------------------
    model: xgboost or sklearn multiclass model

    df_training: Pandas Dataframe
    Training set dataframe

    y_training: array
    Training set labels. The candidates for each
    class should be labeled with 0, ..., N.

    n_classes: int
    Number of classes: should be greater than two. Otherwise
    use the standard cross_val_score(sklearn) implementation

    nfold: int
    Number of folds to calculate the cross
    validation error

    Output
    ------------------------------------------------
    mean: int
    Average of the scores evaluated in the k-different folds


    """
    scores = []
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for index in kfold.split(training_df):
        x_train = training_df.iloc[index[0]]
        y_train = y_training[index[0]]
        x_test = training_df.iloc[index[1]]
        y_test = y_training[index[1]]
        model.fit(x_train, y_train)
        y_score = model.predict_proba(x_test)
        y_test_multi = label_binarize(y_test, classes=range(n_classes))
        score = roc_auc_score(y_test_multi, y_score, average='micro')
        scores.append(score)
    mean = np.mean(scores)
    return mean
