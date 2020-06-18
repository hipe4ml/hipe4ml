"""
Module used to test the model parameter conversion funtionality
"""

import numpy as np
import xgboost as xgb

from hipe4ml.model_handler import ModelHandler


def test_param_conversion():
    """
    Test the model parameter type conversion funtionality
    """
    init_dict = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1., 'colsample_bynode': 1.,
                 'colsample_bytree': 1., 'gamma': 1., 'learning_rate': 1., 'max_delta_step': 0, 'max_depth': 3,
                 'min_child_weight': 1, 'missing': np.nan, 'n_estimators': 100, 'n_jobs': 1,
                 'objective': 'multi:softprob', 'random_state': 0, 'reg_alpha': 1., 'reg_lambda': 1.,
                 'scale_pos_weight': 1., 'subsample': 1., 'verbosity': 1}

    orig_dict = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 0.72, 'colsample_bynode': 0.81,
                 'colsample_bytree': 0.94, 'gamma': 5.5, 'learning_rate': 0.012, 'max_delta_step': 0.3,
                 'max_depth': 6.87, 'min_child_weight':  6.22, 'missing': np.nan, 'n_estimators': 1127.9, 'n_jobs': 1,
                 'objective': 'multi:softprob', 'random_state': 0, 'reg_alpha': 0.4, 'reg_lambda': 2.2,
                 'scale_pos_weight': 11.4, 'subsample': 0.91, 'verbosity': 1}

    right_dict = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 0.72, 'colsample_bynode': 0.81,
                  'colsample_bytree': 0.94, 'gamma': 5.5, 'learning_rate': 0.012, 'max_delta_step': 0,
                  'max_depth': 7, 'min_child_weight': 6, 'missing': np.nan, 'n_estimators': 1128, 'n_jobs': 1,
                  'objective': 'multi:softprob', 'random_state': 0, 'reg_alpha': 0.4, 'reg_lambda': 2.2,
                  'scale_pos_weight': 11.4, 'subsample': 0.91, 'verbosity': 1}

    model = ModelHandler(xgb.XGBClassifier(), None, init_dict)
    converted_dict = model._ModelHandler__cast_model_params(orig_dict)  # pylint: disable=protected-access
    assert converted_dict == right_dict, 'Wrong conversion of model parameters!'
