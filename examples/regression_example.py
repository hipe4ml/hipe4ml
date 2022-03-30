"""
Minimal example to run the regression methods
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from hipe4ml.model_handler import ModelHandler

N_JOBS = 4  # set number of jobs to be executed in parallel

df = pd.DataFrame({'x': np.linspace(0, 2*np.pi, 1000)})
y = np.sin(df['x']) + np.random.normal(0, 0.1, 1000)
TRAIN_SET, TEST_SET, Y_TRAIN, Y_TEST = train_test_split(
    df, y, test_size=0.5, random_state=42)
DATA = [TRAIN_SET, Y_TRAIN, TEST_SET, Y_TEST]
# --------------------------------------------

# TRAINING AND TESTING
# --------------------------------------------
INPUT_MODEL = lgb.LGBMRegressor(n_jobs=N_JOBS)
MODEL = ModelHandler(INPUT_MODEL, task_type='regression')

print("Starting optuna optimisation ...")
hyper_pars_ranges = {'n_estimators': (20, 300), 'max_depth': (
    2, 6), 'learning_rate': (0.01, 0.1)}

rnd_study = MODEL.optimize_params_optuna(DATA, hyper_pars_ranges, cross_val_scoring='neg_mean_absolute_error',
                                         timeout=60, n_trials=100, direction='maximize')
print("Training the final model ...")

y_pred = MODEL.train_test_model(DATA, return_prediction=True)

sorted_indexes = np.argsort(np.array(TEST_SET['x']))
x_sorted = np.array(TEST_SET['x'])[sorted_indexes]
y_sorted = np.array(Y_TEST)[sorted_indexes]
plt.plot(x_sorted, y_sorted, 'b.')
plt.plot(x_sorted, y_pred[sorted_indexes], 'r')
plt.show()
