"""
Module containing the class used for wrapping the models from different
ML libraries to build a new model with common methods
"""
import inspect
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


class ModelHandler:
    """
    Class used for wrapping the models from different ML libraries to
    build a new model with common methods

    Input
    -------------------------------------------------
    input_model: xgboost or sklearn model

    training_columns: list
    Contains the name of the features used for the training.
    Example: ['dEdx', 'pT', 'ct']

    model_params: dict
    Model (hyper-)parameter values. For
    example: silent, n_jobs, objective,
    random_state, eval_metric, tree_method,
    max_depth, learning_rate,
    n_estimators, gamma, min_child_weight,
    subsample, colsample_bytree

    """

    def __init__(self, input_model, training_columns=None, model_params=None):
        self.model = input_model
        self.model_string = inspect.getmodule(
            self.model).__name__.partition('.')[0]
        self.training_columns = training_columns
        self.model_params = model_params
        if model_params is not None:
            self.model_params = model_params
        else:
            self.model_params = self.model.get_params()

    def set_model_params(self, model_params):
        """
        Set the model (hyper-)parameters

        Input
        ------------------------------------
        model_params: dict
        Model (hyper-)parameter values. For
        example: silent, n_jobs, objective,
        random_state, eval_metric, tree_method,
        max_depth, learning_rate,
        n_estimators, gamma, min_child_weight,
        subsample, colsample_bytree

        """
        self.model_params = model_params
        self.model.set_params(**model_params)

    def get_model_params(self):
        """
        Get the model (hyper-)parameters

        Output
        ------------------------------------
        dict
        Model (hyper-)parameter values. For
        example: silent, n_jobs, objective,
        random_state, eval_metric, tree_method,
        max_depth, learning_rate,
        n_estimators, gamma, min_child_weight,
        subsample, colsample_bytree

        """
        return self.model.get_params()

    def set_training_columns(self, training_columns):
        """
        Set the features used for the training process
        Input
        ------------------------------------
        training_columns: list
        Contains the name of the features used for the training.
        Example: ['dEdx', 'pT', 'ct']

        """
        self.training_columns = training_columns

    def get_training_columns(self):
        """
        Get the features used for the training process
        Output
        ------------------------------------
        list
        Contains the name of the features used for the training.
        Example: ['dEdx', 'pT', 'ct']

        """

        return self.training_columns

    def get_original_model(self):
        """
        Get the original unwrapped model

        Output
        ---------------------------
        sklearn or xgboost model
        """
        return self.model

    def get_model_module(self):
        """
        Get the string containing the name
        of the model module

        Output
        ---------------------------
        str
        Name of the model module
        """
        return self.model_string

    def fit(self, x_train, y_train):
        """
        Fit Model

        Input
        ---------------------------
        x_train: array-like, sparse matrix
        Training data

        y_train:array-like, sparse matrix
        Target data

        """
        if self.training_columns is not None:
            self.model.fit(x_train[self.training_columns], y_train)
        else:
            self.model.fit(x_train, y_train)

    def predict(self, x_test, output_margin=True):
        """
        Return model prediction for the array x_test

        Input
        --------------------------------------
        x_test: array-like, sparse matrix
        The input samples. Internally, its dtype
        will be converted to dtype=np.float32. If a
        sparse matrix is provided, it will be converted
        to a sparse csr_matrix.

        output_margin: bool
        Whether to output the raw untransformed margin value.

        Output
        ---------------------------------------
        numpy array
        Model prediction

        """
        if self.training_columns is not None:
            x_test = x_test[self.training_columns]

        if self.model_string == 'xgboost':
            pred = self.model.predict(x_test, output_margin=output_margin)
        if self.model_string == 'sklearn':
            if output_margin:
                pred = self.model.decision_function(x_test).ravel()
            else:
                pred = self.model.predict_proba(x_test)[:, 1]

        return pred

    def train_test_model(self, data):
        """
        Perform the training and the testing of the model

        Input
        ----------------------------------------------

        data: list
        Contains respectively: training
        set dataframe, training label array,
        test set dataframe, test label array

        Output
        ---------------------------------------------
        sklearn or xgboost trained model
        """

        # final training with the optimized hyperparams
        print('Training the final model: ...', end='\r')
        self.fit(data[0], data[1])
        print('{}'.format(self.get_model_params()))
        print('Training the final model: Done!\n')
        print('Testing the model: ...', end='\r')
        y_pred = self.predict(data[2], output_margin=False)
        roc_score = roc_auc_score(data[3], y_pred)
        print('Testing the model: Done!\n')

        print('ROC_AUC_score: {}\n'.format(roc_score))
        print('==============================\n')

    def evaluate_hyperparams(self, data, opt_params, metrics, nfold=5):
        """
        Calculate for a set of hyperparams the cross val score

        Input
        ------------------------------------------

        data: list
        Contains respectively: training
        set dataframe, training label array,
        test set dataframe, test label array

        opt_params: dict
        Hyperparameters to be optimized. For
        example: max_depth, learning_rate,
        n_estimators, gamma, min_child_weight,
        subsample, colsample_bytree

        metrics: string, callable or None
        A string (see sklearn model evaluation
        documentation:
        https://scikit-learn.org/stable/modules/model_evaluation.html)
        or a scorer callable object / function
        with signature scorer(estimator, X, y)
        which should return only a
        single value.

        nfold: int
        Number of folds to calculate the cross
        validation error

        Output
        ---------------------------------------------------------
        float
        Cross validation score evaluated using
        the ROC AUC metrics


        """
        opt_params = self.cast_model_params(opt_params)
        params = {**self.model_params, **opt_params}
        self.model.set_params(**params)
        if self.training_columns is not None:
            return np.mean(cross_val_score(self.model, data[0][self.training_columns], data[1],
                                           cv=nfold, scoring=metrics))

        return np.mean(cross_val_score(self.model, data[0], data[1], cv=nfold, scoring=metrics))

    def optimize_params_bayes(self, data, hyperparams_ranges, metrics, nfold=5, init_points=5,
                              n_iter=5):
        """
        Perform Bayesian optimization

        Input
        ------------------------------------------------------

        data: list
        Contains respectively: training
        set dataframe, training label array,
        test set dataframe, test label array

        hyperparams_ranges: dict
        Xgboost hyperparameter ranges(in tuples).
        For example:
        dict={
            'max_depth':(10,100)
            'learning_rate': (0.01,0.03)
        }

        metrics: string, callable or None
        A string (see sklearn model evaluation
        documentation:
        https://scikit-learn.org/stable/modules/model_evaluation.html)
        or a scorer callable object / function
        with signature scorer(estimator, X, y)
        which should return only a
        single value.

        nfold: int
        Number of folds to calculate the cross validation error

        init_points: int
        How many steps of random exploration you want to perform.
        Random exploration can help by diversifying the exploration space


        Output
        ---------------------------------------------------------
        dict
        Contains the optimized parameters

        """

        # just an helper function
        def hyperparams_crossvalidation(**kwargs):
            return self.evaluate_hyperparams(data, kwargs, metrics, nfold)
        print('')

        optimizer = BayesianOptimization(f=hyperparams_crossvalidation,
                                         pbounds=hyperparams_ranges, verbose=2, random_state=42)
        optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='poi')
        print('')

        # extract and show the results of the optimization
        max_params = {key: None for key in hyperparams_ranges.keys()}
        for key in max_params.keys():
            max_params[key] = optimizer.max['params'][key]
        print('Best target: {0:.6f}'.format(optimizer.max['target']))
        print('Best parameters: {}'.format(max_params))
        self.set_model_params(
            {**self.model_params, **self.cast_model_params(max_params)})

    def cast_model_params(self, params):
        """
        Check if each model parameter is defined as integer
        and change the parameter dictionary consequently

        Input
        -----------------------------------------------------

        params: dict
        Hyperparameter values. For
        example: max_depth, learning_rate,
        n_estimators, gamma, min_child_weight,
        subsample, colsample_bytree

        Output:
        -----------------------------------------------------
        dict
        Hyperparameter values updated

        """
        for key in params.keys():
            if isinstance(self.model.get_params()[key], int):
                params[key] = int(round(params[key]))
        return params
