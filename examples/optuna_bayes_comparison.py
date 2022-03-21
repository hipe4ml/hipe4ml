import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml import plot_utils
import time

if not os.path.exists(f'downloads'):
      os.makedirs(f'downloads')
      print("Downloading test dataset...")
      os.system('curl -L https://cernbox.cern.ch/index.php/s/PBtQEyBt9zFJ9Fx/download --output data.root')
      os.system('curl -L https://cernbox.cern.ch/index.php/s/98tSXndX0VhUMbi/download --output prompt.root')

promptH = TreeHandler('downloads/prompt.root','treeMLDplus')
dataH = TreeHandler('downloads/data.root','treeMLDplus')
bkgH = dataH.get_subset('inv_mass < 1.82 or 1.92 < inv_mass < 2.00', size=promptH.get_n_cand()*3)


train_test_data = train_test_generator([promptH, bkgH], [1,0], test_size=0.5, random_state=42)


vars_to_draw = promptH.get_var_names()
leg_labels = ['background', 'signal']
features_for_train = vars_to_draw.copy()
features_for_train.remove('inv_mass')
features_for_train.remove('pt_cand')


model_clf = xgb.XGBClassifier(verbosity=0)
model_hdl = ModelHandler(model_clf, features_for_train)
model_hdl.set_model_params({'n_jobs':1,'objective': 'binary:logistic','use_label_encoder': False})

hyper_pars_ranges = {'n_estimators': (20, 100), 'max_depth': (2, 4), 'learning_rate': (0.01, 0.1)}

start = time.time()
optuna_study = model_hdl.optimize_params_optuna(train_test_data, hyper_pars_ranges, scoring = 'roc_auc', 
                                                    timeout = 60, n_jobs = 10, n_trials = None, direction='maximize')
optune_time = time.time() - start

trials_optuna = optuna_study.trials
trials_array_optuna = np.array([t.values[0] for t in trials_optuna])

start = time.time()
bayes_study = model_hdl.optimize_params_bayes(train_test_data, hyper_pars_ranges, 'roc_auc', nfold=5, 
                                                init_points=20, n_iter=len(trials_optuna), njobs=1)
bayes_time = time.time() - start


trials_bayes = [trial['target'] for trial in bayes_study.res]

T = 10
mask = np.array([i/T for i in range(T)])

plt.plot(trials_array_optuna, 'o', label = 'Optuna Trials', alpha = 0.2)
optuna_ma = [np.sum(trials_array_optuna[i-T:i]*mask)/np.sum(mask) for i in range(T, len(trials_array_optuna))]
plt.plot(np.array(list(range(len(optuna_ma)))) + T, optuna_ma, label = 'Optuna MA', color = 'black')

plt.plot(trials_bayes, 'o', label = 'Optuna Trials', alpha = 0.2)
bayes_ma = [np.sum(trials_bayes[i-T:i]*mask)/np.sum(mask) for i in range(T, len(trials_bayes))]
plt.plot(np.array(list(range(len(bayes_ma)))) + T, bayes_ma, label = 'Optuna MA', color = 'black')


#plt.plot(np.maximum.accumulate(trials_array), label = 'Optuna Best')
#plt.plot(np.ones(len(trials_array)) * 0.994424, label = 'Default XGBOOST', color = 'red')
#plt.plot(np.ones(len(trials_array)) * 0.99313, label = 'PbPb hyperparameters')
#plt.plot(bayes, 'o', label = 'Bayes trials', alpha = 0.2)
#bayes_ma = [np.sum(bayes[i-T:i]*mask)/np.sum(mask) for i in range(T, len(bayes))]
#plt.plot(np.array(list(range(len(bayes_ma)))) + T, bayes_ma, label = 'Bayes MA', color = 'red')
#plt.plot(np.maximum.accumulate(bayes), label = 'Bayes best')


plt.title('Comparison of hyperparameters optimizers')
plt.xlabel('Iteration')
plt.ylabel('ROC AUC')
plt.legend()

print('Optuna -- time:', optune_time, 'best result:', max(optuna_ma))
print('Bayes  -- time:', bayes_time, 'best result:', max(bayes_ma))

plt.savefig('comparison.png')




