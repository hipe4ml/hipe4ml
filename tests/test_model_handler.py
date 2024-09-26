"""
Module used to test the ModelHandler class functionalities
"""
import pickle
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from optuna.samplers import TPESampler
import helpers as hp
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler
from hipe4ml.analysis_utils import train_test_generator


# globals
SEED = 42

test_data, reference_list = hp.init_handlers_test_workspace(
    Path(__file__).resolve().parent, 'model_handler')
with open(reference_list[0], 'rb') as handle:
    bin_classification_reference = pickle.load(handle)
with open(reference_list[1], 'rb') as handle:
    multi_classification_reference = pickle.load(handle)
with open(reference_list[2], 'rb') as handle:
    regression_reference = pickle.load(handle)


bkg_hdlr = TreeHandler(test_data[0], 'treeMLDplus')
prompt_hdlr = TreeHandler(test_data[1], 'treeMLDplus')
feed_down_hdlr = TreeHandler(test_data[2], 'treeMLDplus')
bkg_hdlr.shuffle_data_frame(size=prompt_hdlr.get_n_cand()*3, random_state=SEED)


training_columns = prompt_hdlr.get_var_names()
training_columns.remove('inv_mass')
training_columns.remove('pt_cand')
train_test_data = train_test_generator(
    [prompt_hdlr, bkg_hdlr], [1, 0], test_size=0.5, random_state=SEED)

train_test_data_multiclass = train_test_generator(
    [feed_down_hdlr, prompt_hdlr, bkg_hdlr], [2, 1, 0], test_size=0.5, random_state=SEED)


def test_training_columns():
    """
    Test training columns setting and getting
    """
    model_hdlr = ModelHandler(XGBClassifier(
        random_state=SEED), 'classification')
    model_hdlr.set_training_columns(training_columns)
    assert model_hdlr.get_training_columns() == training_columns


def test_task_type():
    """
    Test task type setting and getting
    """
    model_hdlr = ModelHandler(XGBClassifier(
        random_state=SEED),  task_type='classification')
    assert model_hdlr.get_task_type() == 'classification'
    model_hdlr = ModelHandler(XGBClassifier(
        random_state=SEED), task_type='regression')
    assert model_hdlr.get_task_type() == 'regression'


def test_model_module():
    """
    Test model module setting and getting
    """
    model_hdlr = ModelHandler(XGBClassifier(
        random_state=SEED), task_type='classification')
    assert model_hdlr.get_model_module() == 'xgboost'
    model_hdlr = ModelHandler(LGBMClassifier(
        random_state=SEED),  task_type='classification')
    assert model_hdlr.get_model_module() == 'lightgbm'
    model_hdlr = ModelHandler(AdaBoostClassifier(
        random_state=SEED),  task_type='classification')
    assert model_hdlr.get_model_module() == 'sklearn'


def test_model_params():
    """
    Test model parameters setting and getting
    """
    model_hdlr = ModelHandler(XGBClassifier(
        random_state=SEED),  task_type='classification')
    model_hdlr.set_model_params({'max_depth': 3})
    assert model_hdlr.get_model_params()['max_depth'] == 3


def test_binary_classification():
    """
    Test binary classification model training and prediction
    """
    classification_model_list = [
        XGBClassifier(use_label_encoder=False), LGBMClassifier(), AdaBoostClassifier()]
    prediction_list = []
    for model in classification_model_list:
        model_hdlr = ModelHandler(model, task_type='classification')
        model_hdlr.set_training_columns(training_columns)
        model_hdlr.set_model_params({'n_estimators': 3, 'random_state': SEED})
        pred = model_hdlr.train_test_model(
            train_test_data, return_prediction=True, output_margin=True)
        prediction_list.append(pred)
    assert len(bin_classification_reference) == len(prediction_list)
    for reference, prediction in zip(bin_classification_reference, prediction_list):
        assert np.allclose(reference, prediction, rtol=1e-5)


def test_multi_classification():
    """
    Test multi classification model training and prediction
    """
    classification_model_list = [
        XGBClassifier(use_label_encoder=False), LGBMClassifier(), AdaBoostClassifier()]
    prediction_list = []
    for model in classification_model_list:
        model_hdlr = ModelHandler(model, task_type='classification')
        model_hdlr.set_training_columns(training_columns)
        model_hdlr.set_model_params({'n_estimators': 3, 'random_state': SEED})
        pred = model_hdlr.train_test_model(
            train_test_data_multiclass, return_prediction=True, output_margin=False, multi_class_opt='ovo')
        prediction_list.append(pred)
    assert len(multi_classification_reference) == len(prediction_list)
    for reference, prediction in zip(multi_classification_reference, prediction_list):
        assert np.allclose(reference, prediction, rtol=1e-5)


def test_regression():
    """
    Test regression model training and prediction
    """
    regression_model_list = [XGBRegressor(), LGBMRegressor(
    ), AdaBoostRegressor()]
    prediction_list = []
    for model in regression_model_list:
        model_hdlr = ModelHandler(model, task_type='regression')
        model_hdlr.set_training_columns(training_columns)
        model_hdlr.set_model_params({'n_estimators': 3, 'random_state': SEED})
        print(model_hdlr.get_model_params())
        pred = model_hdlr.train_test_model(
            train_test_data, return_prediction=True)
        prediction_list.append(pred)
    assert len(regression_reference) == len(prediction_list)
    for reference, prediction in zip(regression_reference, prediction_list):
        assert np.allclose(reference, prediction, rtol=1e-5)


def test_hyperparams_optimization():
    """
    Test hyperparameter optimization
    """
    n_estimators_ref = 135
    max_depth_ref = 2

    hyper_pars_ranges = {'n_estimators': (10, 200), 'max_depth': (
        2, 6)}
    model_hdlr = ModelHandler(XGBClassifier(), task_type='classification')
    model_hdlr.set_training_columns(training_columns)
    model_hdlr.set_model_params(
        {'n_jobs': 1, 'objective': 'binary:logistic', 'use_label_encoder': False, 'random_state': SEED})
    model_hdlr.optimize_params_optuna(train_test_data, hyper_pars_ranges,
                                      cross_val_scoring='roc_auc', n_trials=20,
                                      direction='maximize', optuna_sampler=TPESampler(seed=SEED))
    assert model_hdlr.get_model_params()['n_estimators'] == n_estimators_ref
    assert model_hdlr.get_model_params()['max_depth'] == max_depth_ref


def test_handler_dump_load():
    """
    Test model handler dump and load
    """
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.joinpath('tmp_test/data/')
    model_hdlr = ModelHandler(XGBClassifier(use_label_encoder=False), task_type='classification')
    model_hdlr.set_training_columns(training_columns)
    model_hdlr.set_model_params({'n_estimators': 3, 'random_state': SEED})
    native_pred = model_hdlr.train_test_model(train_test_data, return_prediction=True)
    model_hdlr.dump_model_handler(f'{data_path}/model_handler.pkl')
    loaded_model_hdlr = ModelHandler()
    loaded_model_hdlr.load_model_handler(f'{data_path}/model_handler.pkl')
    loaded_pred = loaded_model_hdlr.predict(train_test_data[2], output_margin=False)
    assert np.allclose(native_pred, loaded_pred, rtol=1e-5)
    assert model_hdlr.get_task_type() == loaded_model_hdlr.get_task_type()
    assert model_hdlr.get_model_module() == loaded_model_hdlr.get_model_module()
    assert model_hdlr.get_training_columns() == loaded_model_hdlr.get_training_columns()
    assert model_hdlr.get_n_classes() == loaded_model_hdlr.get_n_classes()


def test_xgb_dump_load():
    """
    Test xgb model dump and load
    """
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.joinpath('tmp_test/data/')

    model_hdlr = ModelHandler(XGBClassifier(use_label_encoder=False), task_type='classification')
    model_hdlr.set_training_columns(training_columns)
    model_hdlr.set_model_params({'n_estimators': 3, 'random_state': SEED})
    native_pred = model_hdlr.train_test_model(train_test_data, return_prediction=True)
    model_hdlr.dump_original_model(f'{data_path}/xgb_model.model', xgb_format=True)
    model_hdlr.dump_original_model(f'{data_path}/xgb_model.pkl', xgb_format=False)

    loaded_model_xgb = XGBClassifier(use_label_encoder=False)
    loaded_model_xgb.load_model(f'{data_path}/xgb_model.model')
    loaded_pred_xgb = loaded_model_xgb.predict_proba(train_test_data[2][training_columns])[:, 1]
    assert np.allclose(native_pred, loaded_pred_xgb, rtol=1e-5)

    with open(f'{data_path}/xgb_model.pkl', 'rb') as model_file:
        loaded_model_pkl = pickle.load(model_file)
    loaded_pred_pkl = loaded_model_pkl.predict_proba(train_test_data[2][training_columns])[:, 1]
    assert np.allclose(native_pred, loaded_pred_pkl, rtol=1e-5)

    hp.terminate_handlers_test_workspace(Path(__file__).resolve().parent)
