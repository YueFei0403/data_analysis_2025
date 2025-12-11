import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text, export_graphviz
import graphviz
import matplotlib.pyplot as plt

from econml.dml import CausalForestDML
from econml.grf import CausalForest
import matplotlib.pyplot as plt
import os, sys
log_path = "./results/output.log"
import warnings
warnings.filterwarnings('ignore')

from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt

from datetime import datetime


def extract_feature_matrices(df):
    X_df = df[FEATURE_COLS].copy()
    Y_arr = df[OUTCOME_COL].astype(int).values.ravel()

    df["group_code"] = df[TREATMENT_COL].astype("category").cat.codes
    T_arr = df["group_codes"].values
    X_arr = preprocessor.fit_transform(X_df)

    return X_arr, Y_arr, T_arr, preprocessor


def choose_strategy(tau1, tau2):
    """
        - if both <= 0 -> control group (including no effect to reduce operational cost)
        - otherwise choose argmax
    """
    if tau1 <= 0 and tau2 <= 0:
        return 0    # Control
    if tau1 > tau2:
        return 1    # T1
    else:   
        return 2    # T2



def compute_qini_curve(ite, y, t_bin):
    """
        Formula:
            tau[X_i] = E[Y_i | W_i = 1] - E[Y_i | W_i = 0] for X_i = x
        For the group of people X_i with similar traits, what the average/expected benefits can be brought by treatment

    """
    ite = np.asarray(ite).flatten()
    y = np.asarray(y).flatten()
    t_bin = np.asarray(t_bin).flatten()

    #  Sort by uplift descending
    df_eval = pd.DataFrame({
        'y': y,
        'T': t_bin,
        'uplift': ite
    }).sort_values('uplift', ascending=False)

    N = len(df_eval)

    # Cumulative treated / control counts
    df_eval['n_treat_cum'] = df_eval['T'].cumsum()
    df_eval['n_control_cum'] = (1 - df_eval['T']).cumsum()

    # Cumulative outcomes
    df_eval['y_treat_cum'] = (df_eval['y'] * df_eval['T']).cumsum()
    df_eval['y_control_cum'] = (df_eval['y'] * (1 - df_eval['T'])).cumsum()