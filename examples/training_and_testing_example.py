"""
Minimal example to run the package utilities
"""
import pandas as pd
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler

# DATA PREPARATION (load data from sklearn digits dataset)
# --------------------------------------------
SKLEARN_DATA = datasets.load_digits(n_class=2)
DIGITS_DATASET = pd.DataFrame(SKLEARN_DATA.data)     # pylint: disable=E1101
Y_DIGITS = SKLEARN_DATA.target       # pylint: disable=E1101
SIG_DF = DIGITS_DATASET[Y_DIGITS == 1]
BKG_DF = DIGITS_DATASET[Y_DIGITS == 0]
TRAIN_SET, TEST_SET, Y_TRAIN, Y_TEST = train_test_split(
    DIGITS_DATASET, Y_DIGITS, test_size=0.5, random_state=42)
DATA = [TRAIN_SET, Y_TRAIN, TEST_SET, Y_TEST]
# --------------------------------------------


# TRAINING AND TESTING
# --------------------------------------------
INPUT_MODEL = xgb.XGBClassifier()
MODEL = ModelHandler(INPUT_MODEL)

# hyperparams optimization
HYP_RANGES = {
    # # defines the maximum depth of a single tree (regularization)
    'max_depth': (5, 15),
    # 'learning_rate': (0.01, 0.3),  # learning rate
    'n_estimators': (5, 10),  # number of boosting trees
}
MODEL.optimize_params_bayes(DATA, HYP_RANGES, 'roc_auc')

# train and test the model with the updated hyperparameters
MODEL.train_test_model(DATA)
Y_PRED = MODEL.predict(DATA[2])

# Calculate the BDT efficiency as a function of the BDT score
EFFICIENCY, THRESHOLD = analysis_utils.bdt_efficiency_array(
    DATA[3], Y_PRED, n_points=10)
# --------------------------------------------


# PLOTTING
# --------------------------------------------
FEATURES_DISTRIBUTIONS_PLOT = plot_utils.plot_distr(
    [SIG_DF, BKG_DF], SIG_DF.columns)
CORRELATION_MATRIX_PLOT = plot_utils.plot_corr([SIG_DF, BKG_DF], SIG_DF.columns)
BDT_OUTPUT_PLOT = plot_utils.plot_output_train_test(MODEL, DATA)
ROC_CURVE_PLOT = plot_utils.plot_roc(DATA[3], Y_PRED)
PRECISION_RECALL_PLOT = plot_utils.plot_precision_recall(DATA[3], Y_PRED)
BDT_EFFICIENCY_PLOT = plot_utils.plot_bdt_eff(THRESHOLD, EFFICIENCY)
FEATURES_IMPORTANCE = plot_utils.plot_feature_imp(TEST_SET, Y_TEST, MODEL)
plt.show()
# ---------------------------------------------
