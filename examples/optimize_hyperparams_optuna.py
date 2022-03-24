
"""
Example for optimizing hypeparams with optuna and evaluate the performance of different samplers
"""
import os
import time
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from optuna.samplers import RandomSampler
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator


def moving_average(array, n_mov=4):
    """
    Compute moving average
    """
    ret = np.cumsum(array, dtype=float)
    ret[n_mov:] = ret[n_mov:] - ret[:-n_mov]
    return ret[n_mov - 1:] / n_mov


if not os.path.exists('downloads'):
    os.makedirs('downloads')
    print("Downloading test dataset...")
if not os.path.exists('downloads/data.root'):
    os.system(
        'curl -L https://cernbox.cern.ch/index.php/s/PBtQEyBt9zFJ9Fx/download --output downloads/data.root')
if not os.path.exists('downloads/prompt.root'):
    os.system('curl -L https://cernbox.cern.ch/index.php/s/98tSXndX0VhUMbi/download --output downloads/prompt.root')

N_JOBS = 40  # set number of jobs to be executed in parallel

promptH = TreeHandler('downloads/prompt.root', 'treeMLDplus')
dataH = TreeHandler('downloads/data.root', 'treeMLDplus')
bkgH = dataH.get_subset(
    'inv_mass < 1.82 or 1.92 < inv_mass < 2.00', size=promptH.get_n_cand()*3)


train_test_data = train_test_generator(
    [promptH, bkgH], [1, 0], test_size=0.5, random_state=42)


features = promptH.get_var_names()
features_for_train = features.copy()
features_for_train.remove('inv_mass')
features_for_train.remove('pt_cand')


model_clf = xgb.XGBClassifier(verbosity=0)
model_hdl = ModelHandler(model_clf, features_for_train)
model_hdl.set_model_params(
    {'n_jobs': 1, 'objective': 'binary:logistic', 'use_label_encoder': False})

hyper_pars_ranges = {'n_estimators': (20, 300), 'max_depth': (
    2, 6), 'learning_rate': (0.01, 0.1)}

print("Starting optuna optimisation w/ Random Sampler")
start = time.time()
rnd_study = model_hdl.optimize_params_optuna(train_test_data, hyper_pars_ranges, cross_val_scoring='roc_auc',
                                             timeout=60, n_jobs=N_JOBS, n_trials=100, direction='maximize',
                                             optuna_sampler=RandomSampler())
rnd_time = time.time() - start
trials_rnd = rnd_study.trials
trials_array_rnd = np.array([t.values[0] for t in trials_rnd])


model_clf = xgb.XGBClassifier(verbosity=0)
model_hdl = ModelHandler(model_clf, features_for_train)
model_hdl.set_model_params(
    {'n_jobs': 1, 'objective': 'binary:logistic', 'use_label_encoder': False})
print("Starting optuna optimisation w/ TPEsampler")
start = time.time()
tpe_study = model_hdl.optimize_params_optuna(train_test_data, hyper_pars_ranges, cross_val_scoring='roc_auc',
                                             timeout=60, n_jobs=N_JOBS, n_trials=100, direction='maximize')
tpe_time = time.time() - start
trials_tpe = tpe_study.trials
trials_array_tpe = np.array([t.values[0] for t in trials_tpe])

tpe_ma = moving_average(trials_array_tpe, 4)
rnd_ma = moving_average(trials_array_rnd, 4)
x_axis_ma = np.arange(len(rnd_ma)) + 4 - 1

plt.plot(trials_array_tpe, 'o',
         label='TPEsampler', alpha=0.2)
plt.plot(x_axis_ma, tpe_ma, label='TPESampler MA', color='blue')

plt.plot(trials_array_rnd, 'o',
         label='Random Sampler Trials', alpha=0.2)
plt.plot(x_axis_ma, rnd_ma, label='Random Sampler MA', color='orange')


plt.title('Comparison of hyperparameters optimizers')
plt.xlabel('Iteration')
plt.ylabel('ROC AUC')
plt.legend()

print('TPESampler -- time:', tpe_time, 'best result:', max(tpe_ma))
print('Random Sampler  -- time:', rnd_time, 'best result:', max(rnd_ma))
plt.show()
