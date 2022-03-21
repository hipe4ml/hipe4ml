
"""
Comparison between optuna and bayes-opt performance
"""
import os
import time
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
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

MOV_AVER_N = 4

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
    2, 5), 'learning_rate': (0.01, 0.1)}

print("Starting optuna optimisation w/ TPEsampler")
start = time.time()
optuna_study = model_hdl.optimize_params_optuna(
    train_test_data, hyper_pars_ranges, cross_val_scoring='roc_auc', timeout=60, n_jobs=10, n_trials=100, direction='maximize')
optune_time = time.time() - start
trials_optuna = optuna_study.trials
trials_array_optuna = np.array([t.values[0] for t in trials_optuna])

print("Starting bayesian optimisation w/ Bayes-opt")
INIT_POINTS = 15
start = time.time()
bayes_study = model_hdl.optimize_params_bayes(train_test_data, hyper_pars_ranges, cross_val_scoring='roc_auc',
                                              nfold=5, init_points=INIT_POINTS, n_iter=len(trials_optuna) - INIT_POINTS, njobs=10)
bayes_time = time.time() - start
trials_bayes = [trial['target'] for trial in bayes_study.res]

optuna_ma = moving_average(trials_array_optuna, MOV_AVER_N)
bayes_ma = moving_average(trials_bayes, MOV_AVER_N)
x_axis_ma = np.arange(len(bayes_ma)) + MOV_AVER_N - 1


plt.plot(trials_array_optuna, 'o',
         label=f'Optuna Trials, elapsed time: {optune_time:.2} s', alpha=0.2)
plt.plot(x_axis_ma, optuna_ma, label='Optuna MA', color='blue')

plt.plot(trials_bayes, 'o',
         label=f'Bayes-opt Trials, elapsed time: {bayes_time:.2} s', alpha=0.2)
plt.plot(x_axis_ma, bayes_ma, label='Bayes-opt MA', color='orange')


plt.title('Comparison of hyperparameters optimizers')
plt.xlabel('Iteration')
plt.ylabel('ROC AUC')
plt.legend()

print('Optuna -- time:', optune_time, 'best result:', max(optuna_ma))
print('Bayes  -- time:', bayes_time, 'best result:', max(bayes_ma))
plt.show()
