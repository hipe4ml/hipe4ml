"""
Module containing the class used for wrapping the models from different
ML libraries to build a new model with common methods
"""
from copy import deepcopy
import inspect
import pickle

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import hipe4ml.tree_handler


class ModelHandler:
    """
    Class used for wrapping the models from different ML libraries to
    build a new model with common methods. Currently only XGBoost
    (through it's sklearn interface) and sklearn models are supported.

    Parameters
    -------------------------------------------------
    input_model: xgboost or sklearn model

    training_columns: list
        Contains the name of the features used for the training.
        Example: ['dEdx', 'pT', 'ct']

    model_params: dict
        Model hyper-parameter values. For
        example (XGBoost): max_depth, learning_rate,
        n_estimators, gamma, min_child_weight, ...
    """

    def __init__(self, input_model=None, training_columns=None, model_params=None):
        self.model = input_model
        self.training_columns = training_columns
        self.model_params = model_params
        self._n_classes = None

        if self.model is not None:
            self.model_string = inspect.getmodule(self.model).__name__.partition('.')[0]

            if self.model_params is None:
                self.model_params = self.model.get_params()
            else:
                self.model.set_params(**self.model_params)

    def set_model_params(self, model_params):
        """
        Set the model (hyper-)parameters

        Parameters
        ------------------------------------
        model_params: dict
            Model hyper-parameter values. For
            example (XGBoost): max_depth, learning_rate,
            n_estimators, gamma, min_child_weight, ...
        """
        self.model_params = model_params
        self.model.set_params(**self.model_params)

    def get_model_params(self):
        """
        Get the model (hyper-)parameters

        Returns
        ------------------------------------
        out: dict
            Model hyper-parameter values. For
            example (XGBoost): max_depth, learning_rate,
            n_estimators, gamma, min_child_weight, ...
        """
        return self.model.get_params()

    def set_training_columns(self, training_columns):
        """
        Set the features used for the training process

        Parameters
        ------------------------------------
        training_columns: list
            Contains the name of the features used for the training.
            Example: ['dEdx', 'pT', 'ct']
        """
        self.training_columns = training_columns

    def get_training_columns(self):
        """
        Get the features used for the training process

        Returns
        ------------------------------------
        out: list
            Names of the features used for the training.
            Example: ['dEdx', 'pT', 'ct']
        """

        return self.training_columns

    def get_original_model(self):
        """
        Get the original unwrapped model

        Returns
        ---------------------------
        out: sklearn or XGBoost model
        """
        return self.model

    def get_model_module(self):
        """
        Get the string containing the name
        of the model module

        Returns
        ---------------------------
        out: str
            Name of the model module
        """
        return self.model_string

    def get_n_classes(self):
        """
        Get the number of classes

        Returns
        ---------------------------
        out: int
            Number of classes
        """
        return self._n_classes

    def fit(self, x_train, y_train, **kwargs):
        """
        Fit Model

        Parameters
        ---------------------------
        x_train: array-like, sparse matrix
            Training data

        y_train: array-like, sparse matrix
            Target data

        **kwargs:
            Extra kwargs passed on to model.fit() method
        """
        n_classes = len(np.unique(y_train))
        self._n_classes = n_classes
        if self.training_columns is None:
            self.training_columns = list(x_train.columns)

        self.model.fit(x_train[self.training_columns], y_train, **kwargs)

    def predict(self, x_test, output_margin=True, **kwargs):
        """
        Return model prediction for the array x_test

        Parameters
        --------------------------------------
        x_test: hipe4ml tree_handler, array-like, sparse matrix
            The input sample.

        output_margin: bool
            Whether to output the raw untransformed margin value. If False model
            probabilities are returned

        **kwargs:
            Extra kwargs passed on to the model prediction function:
            - predict() (xgboost) or decision_function() (sklearn) if output_margin==True
            - predict_proba() if output_margin==False


        Returns
        ---------------------------------------
        out: numpy array
            Model predictions
        """
        if isinstance(x_test, hipe4ml.tree_handler.TreeHandler):
            x_test = x_test.get_data_frame()

        x_test = x_test[self.training_columns]

        if output_margin:
            if self.model_string == 'xgboost':
                pred = self.model.predict(x_test, True, **kwargs)
            if self.model_string == 'sklearn':
                pred = self.model.decision_function(x_test, **kwargs).ravel()
        else:
            pred = self.model.predict_proba(x_test, **kwargs)
            # in case of binary classification return only the scores of
            # the signal class
            if pred.shape[1] <= 2:
                pred = pred[:, 1]

        return pred

    def train_test_model(self, data, return_prediction=False, output_margin=False, average='macro',
                         multi_class_opt='raise', **kwargs):
        """
        Perform the training and the testing of the model. The model performance is estimated
        using the ROC AUC metric

        Parameters
        ----------------------------------------------
        data: list
            Contains respectively: training
            set dataframe, training label array,
            test set dataframe, test label array

        return_prediction: bool
            If True Model predictions on the test set are
            returned

        output_margin: bool
            Whether to output the raw untransformed margin value. If False model
            probabilities are returned

        average: string
            Option for the average of ROC AUC scores used only in case of multi-classification.
            You can choose between 'macro' and 'weighted'. For more information see
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

        multi_class_opt: string
            Option to compute ROC AUC scores used only in case of multi-classification.
            The one-vs-one 'ovo' and one-vs-rest 'ovr' approaches are available

        **kwargs: dict
            Extra kwargs passed on to the model fit method

        Returns
        ---------------------------------------
        out: numpy array or None
            If return_prediction==True, Model predictions on the test set are
            returned

        """

        # get number of classes
        n_classes = len(np.unique(data[1]))
        self._n_classes = n_classes
        print('Number of detected classes:', n_classes)

        # final training with the optimized hyperparams
        print('Training the final model: ...', end='\r')
        self.fit(data[0], data[1], **kwargs)
        print('Training the final model: Done!')
        print('Testing the model: ...', end='\r')
        y_pred = self.predict(data[2], output_margin=output_margin)
        roc_score = roc_auc_score(
            data[3], y_pred, average=average, multi_class=multi_class_opt)
        print('Testing the model: Done!')

        print(f'ROC_AUC_score: {roc_score:.6f}')
        print('==============================')
        if return_prediction:
            return y_pred
        return None

    def optimize_params_optuna(self, data, hyperparams_ranges, cross_val_scoring, nfold=5, direction='maximize',
                               optuna_sampler=None, resume_study=None, save_study=None, **kwargs):
        """
        Perform hyperparameter optimization of ModelHandler using the Optuna module.
        The model hyperparameters are automatically set as the ones that provided the
        best result during the optimization.

        Parameters
        ------------------------------------------------------
        data: list
            Contains respectively: training
            set dataframe, training label array,
            test set dataframe, test label array

        hyperparams_ranges: dict
            Hyperparameter ranges (in tuples or list). If a parameter is not
            in a tuple or a list it will be considered constant.
            Important: the type of the params must be preserved
            when passing the ranges.
            For example:
            dict={
                'max_depth':(10,100)
                'learning_rate': (0.01,0.03)
                'n_jobs': 8
            }

        cross_val_scoring: string, callable or None
            Score metrics used for the cross-validation.
            A string (see sklearn model evaluation documentation:
            https://scikit-learn.org/stable/modules/model_evaluation.html)
            or a scorer callable object / function with signature scorer(estimator, X, y)
            which should return only a single value.
            In binary classification 'roc_auc' is suggested.
            In multi-classification one between ‘roc_auc_ovr’, ‘roc_auc_ovo’,
            ‘roc_auc_ovr_weighted’ and ‘roc_auc_ovo_weighted’ is suggested.
            For more information see
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

        direction: str
            The direction of optimization. Either 'maximize' or 'minimize'.
            (e.g. for the metric 'roc_auc' the direction is 'maximize')

        optuna_sampler: optuna.samplers.BaseSampler
            Sampler to be used for the optuna (maxi-)minimisation.
            If None, default TPESampler is used. For more information see:
            https://optuna.readthedocs.io/en/stable/reference/samplers.html

        nfold: int
            Number of folds to calculate the cross validation error

        resume_study: str
            A string indicating the filename of the study to be resumed.
            If None, the study is not resumed.

        save_study: str
            A string indicating the filename of the study. If None,
            the study is not saved into a file.

        **kwargs: dict
            Optuna study parameters

        Returns
        ------------------------------------------------------

        study: optuna.study.Study
            The obtuna object which stores the whole study. See Optuna's documentation for more details:
            https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study
        """

        n_classes = len(np.unique(data[1]))
        self._n_classes = n_classes
        if self.training_columns is None:
            self.training_columns = list(data[0].columns)

        x_train, y_train, _, _ = data

        def __get_int_or_uniform(hyperparams_ranges, trial):

            params = {}

            for key in hyperparams_ranges:
                if isinstance(hyperparams_ranges[key][0], int):
                    params[key] = trial.suggest_int(key, hyperparams_ranges[key][0], hyperparams_ranges[key][1])
                elif isinstance(hyperparams_ranges[key][0], float):
                    params[key] = trial.suggest_uniform(key, hyperparams_ranges[key][0], hyperparams_ranges[key][1])

            return params

        def __objective(trial):

            params = __get_int_or_uniform(hyperparams_ranges, trial)
            model_copy = deepcopy(self.model)
            model_copy.set_params(**{**self.model_params, **params})
            return np.mean(cross_val_score(model_copy, x_train[self.training_columns], y_train,
                                           cv=nfold, scoring=cross_val_scoring, n_jobs=1))
        if resume_study:
            with open(resume_study, 'rb') as resume_study_file:
                study = pickle.load(resume_study_file)
        else:
            study = optuna.create_study(direction=direction, sampler=optuna_sampler)

        study.optimize(__objective, **kwargs)

        if save_study:
            with open(save_study, 'wb') as study_file:
                pickle.dump(study, study_file)

        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        best_trial = study.best_trial

        print(f"Value: {best_trial.value}")
        print("Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        self.set_model_params({**self.model_params, **best_trial.params})

        return study

    def dump_original_model(self, filename, xgb_format=False):
        """
        Save the trained model into a pickle
        file. Only for xgboost models it is also given
        the possibility to save them into a .model file

        Parameters
        -----------------------------------------------------
        filename: str
            Name of the file in which the model is saved

        xgb_format : bool
            If True saves the xgboost model into a .model file
        """
        if xgb_format is False:
            with open(filename, "wb") as output_file:
                pickle.dump(self.model, output_file)
        else:
            if self.model_string == 'xgboost':
                self.model.save_model(filename)
            else:
                print("File not saved: only xgboost models support the .model extension")

    def dump_model_handler(self, filename):
        """
        Save the model handler into a pickle file

        Parameters
        -----------------------------------------------------
        filename: str
            Name of the file in which the model is saved
        """
        with open(filename, "wb") as output_file:
            pickle.dump(self, output_file)

    def load_model_handler(self, filename):
        """
        Load a model handler saved into a pickle file

        Parameters
        -----------------------------------------------------
        filename: str
            Name of the file in which the model is saved
        """
        with open(filename, "rb") as input_file:
            loaded_model = pickle.load(input_file)
            self.model = loaded_model.get_original_model()
            self.training_columns = loaded_model.get_training_columns()
            self.model_params = loaded_model.get_model_params()
            self.model.set_params(**self.model_params)
            self.model_string = loaded_model.get_model_module()
            self._n_classes = loaded_model.get_n_classes()
