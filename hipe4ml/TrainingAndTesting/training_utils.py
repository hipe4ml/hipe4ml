""" Module containing the training utils.
    """
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization


def evaluate_hyperparams(data, training_columns, reg_params, hyp_params, nfold=5):
    '''
    Calculate for a set of hyperparams the cross val score

    Input
    --------------------------------------------------------
    data: data array containing respectively: training set, training label, test set, test label

    reg_params: dictionary containing the xgboost regularization values.
    These parameters will be NOT optimized.
    For example: silent, n_jobs, objective, random_state, eval_metric, tree_method

    hyp_params: dictionary containing the xgboost hyperparameter values.
    For example: max_depth, learning_rate, n_estimators, gamma,
    min_child_weight, subsample, colsample_bytree

    nfold: number of folds to calculate the cross validation error

    Output
    ---------------------------------------------------------
    cross val score


    '''

    params = {**reg_params, **hyp_params}

    model = xgb.XGBClassifier(**params)
    return np.mean(cross_val_score(model, data[0][training_columns], data[1], cv=nfold, scoring='roc_auc'))


def optimize_params_bayes(data, training_columns, reg_params, hyperparams_ranges, nfold=3, init_points=5,
                          n_iter=5):
    '''
    Perform Bayesian optimization

    Input
    ------------------------------------------------------

    data: data array containing respectively: training set, training label, test set, test label

    reg_params: dictionary containing the xgboost regularization values.
    These parameters will be NOT optimized.
    For example: silent, n_jobs, objective, random_state, eval_metric, tree_method

    hyperparams_ranges: dictionary containing the xgboost hyperparameter RANGES.
    For example: max_depth, learning_rate, n_estimators, gamma,
    min_child_weight, subsample, colsample_bytree

    nfold: number of folds to calculate the cross validation error

    init_points: number of initial points

    Output
    ---------------------------------------------------------
    max_params: dict containing the optimized parameters

    '''

    # just an helper function
    def hyperparams_crossvalidation(hyp_params):
        return evaluate_hyperparams(
            data, training_columns, reg_params, hyp_params, nfold)

    print('')

    optimizer = BayesianOptimization(f=hyperparams_crossvalidation,
                                     pbounds=hyperparams_ranges, verbose=2, random_state=42)
    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='poi')
    print('')

    # extract and show the results of the optimization
    max_params = {
        'max_depth': int(optimizer.max['params']['max_depth']),
        'learning_rate': optimizer.max['params']['learning_rate'],
        'n_estimators': int(optimizer.max['params']['n_estimators']),
        'gamma': optimizer.max['params']['gamma'],
        'min_child_weight': int(optimizer.max['params']['min_child_weight']),
        'subsample': optimizer.max['params']['subsample'],
        'colsample_bytree': optimizer.max['params']['colsample_bytree'],
        # 'lambda': optimizer.max['params']['lambda'],
        # 'alpha': optimizer.max['params']['alpha'],
    }
    print('Best target: {0:.6f}'.format(optimizer.max['target']))
    print('Best parameters: {}'.format(max_params))

    return max_params


def train_test_model(data, training_columns, reg_params, hyp_params=0,
                     optimize=False, hyperparams_ranges=0):
    '''
    Perform the training and the testing of the model

    Input
    ------------------------------------------------------------

    data: data array containing respectively: training set, training label, test set, test label

    training_columns: training columns

    reg_params: dictionary containing the xgboost regularization values.
    These parameters will be NOT optimized.
    For example: silent, n_jobs, objective, random_state, eval_metric, tree_method

    hyp_params: dictionary containing the xgboost hyperparameter default values.
    For example: max_depth, learning_rate, n_estimators, gamma,
    min_child_weight, subsample, colsample_bytree

    optimize: If True enable the bayesian optimization

    hyperparams_ranges: dictionary containing the xgboost hyperparameter RANGES. Provide It
    only if you want to optimize the model
    For example: max_depth, learning_rate, n_estimators, gamma,
    min_child_weight, subsample, colsample_bytree

    Output
    -----------------------------------------------------------------
    model: trained model



    '''

    data_train = [data[0], data[1]]

    # manage the optimization process
    if optimize:
        print('Hyperparameters optimization: ...', end='\r')
        max_params = optimize_params_bayes(
            data_train, training_columns, reg_params, hyperparams_ranges, init_points=10, n_iter=10)
        print('Hyperparameters optimization: Done!\n')
    else:  # manage the default params
        max_params = hyp_params

    # join the dictionaries of the regressor params with the maximized hyperparams
    best_params = {**max_params, **reg_params}

    # final training with the optimized hyperparams
    print('Training the final model: ...', end='\r')
    model = xgb.XGBClassifier(**best_params)
    model.fit(data[0][training_columns], data[1])
    print('{}'.format(model.get_params()))
    print('Training the final model: Done!\n')

    # BDT output distributions plot

    # test the model performances
    print('Testing the model: ...', end='\r')
    y_pred = model.predict(data[2][training_columns])
    roc_score = roc_auc_score(data[3], y_pred)
    print('Testing the model: Done!\n')

    print('ROC_AUC_score: {}\n'.format(roc_score))
    print('==============================\n')

    return model


def bdt_efficiency_array(df_in, n_points=50):
    """
    Calculate the BDT efficiency as a function of a score threshold

    Input
    ------------------------------------------------
    df_in: labeled test set dataframe
    n_points: length of the efficiency array

    Output
    ------------------------------------------------
    efficiency: efficiency array as a function of the
    threshold value
    threshold: threshold values array


    """
    min_score = df_in['Score'].min()
    max_score = df_in['Score'].max()

    threshold = np.linspace(min_score, max_score, n_points)

    efficiency = []

    n_sig = sum(df_in['y'])

    for thr in threshold:  # pylint: disable=unused-variable
        df_selected = df_in.query('Score>@thr')['y']
        sig_selected = np.sum(df_selected)
        efficiency.append(sig_selected / n_sig)

    return efficiency, threshold
