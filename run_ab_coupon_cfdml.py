import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from econml.dml import CausalForestDML
from econml.orf import DMLOrthoForest, DROrthoForest
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper

from scipy.stats import skew, kurtosis
import os, sys

from cf_helpers import (
    build_cf_result_lite,
    save_cf_result_lite,
    plot_ite_histogram,
    print_ite_analysis
)

#=================================================
#                   HELPER FUNCTIONS       
#=================================================
def add_hist_percentage_labels(counts, bins, patches, fontsize=9, rotation=90, buffer_ratio=0.7):
    """
    Add percentage labels to histogram bars.

    Parameters:
        counts   : histogram counts (returned from plt.hist)
        bins     : bin edges (returned from plt.hist)
        patches  : bar containers (returned from plt.hist)
        fontsize : text font size
        rotation : rotation of the percentage text
    """
    total = sum(counts)
    max_count = max(counts)

    for count, x_left, patch in zip(counts, bins[:-1], patches):
        if count > 0:  # skip empty bins
            x_center = x_left + (bins[1] - bins[0]) / 2
            pct = (count / total) * 100

            # Vertical offset for nicer spacing
            y_pos = count + max_count * buffer_ratio
            
            plt.text(
                x_center,
                count,
                f"{pct:.1f}%",
                ha='center',
                va='bottom',
                fontsize=fontsize,
                rotation=rotation
            )

log_path = "./results/output.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# If old log exists, remove it and print a message
if os.path.exists(log_path):
    os.remove(log_path)
    print(f"[INFO] Old log file removed: {log_path}")
else:
    print(f"[INFO] No previous log file found at: {log_path}")

# Redirect stdout and stderr to a fresh file
sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout  # optional — log errors too
print(f"[INFO] Logging initialized → {log_path}")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
# === Step 1 ===
df = pd.read_csv("user_data.csv")
print(df.describe())

le = LabelEncoder()

df['User_os_encoded'] = le.fit_transform(df['User_os'])
os_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Encoding mapping:, os_mapping")

print(df.head())

# === Step 2 ===
# Tag
C = ["Userid"]

# 特征变量
X = [
    "User_os_encoded",
    "90days_purchase_time",
    "90days_per_purchase_price",
    "90days_purchase_amount",
    "90days_coupon_time",
    "Last_purchase_day",
    "90days_coupon_ratio"
]

# 处理变量 (Treatment)
T = ["group"]

# 结果变量 （Outcome)
Y = ["is_add"]

# === Step 3 ===
df_X = df[X]
df_Y = df[Y]
df_T = df[T]

# === Step 4 ===
# User_os is categorical, others numeric
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), [
        '90days_purchase_time',
        '90days_per_purchase_price',
        '90days_purchase_amount',
        '90days_coupon_time',
        'Last_purchase_day'
    ])
])

# Convert treatment group to integer labels: e.g. 对照组=0, 实验组1=1, 实验组2=2
df["group_code"] = df["group"].astype('category').cat.codes
print(df[["group", "group_code"]].drop_duplicates())

# === Step 5 ===
X_arr = df_X.values            
Y_arr = df_Y.values.ravel()    
T_arr = df["group_code"].values                                    

print("\n>>> Shapes before split: >>>")
print("X:", df_X.shape)
print("Y:", df_Y.shape)
print("T:", np.shape(T_arr))
X_train, X_test, Y_train, Y_test, T_train, T_test = train_test_split(
    X_arr, Y_arr, T_arr, test_size=0.3, random_state=42
)

# To better understand how balanced my experimental/causal-inference setup is:
print("\n>>> Checking Treatment Class Balance >>>")
unique, counts = np.unique(T_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Treatment {u}: {c} samples ({c / len(T_train):.2%})")

# ==============================================
#               Compare T0 and T1
# ==============================================
mask_01 = (T_train == 0) | (T_train == 1)
cf_01 = CausalForestDML(
    model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
    model_y=RandomForestClassifier(n_estimators=100, max_depth=5),
    n_estimators=100,
    min_samples_leaf=20,
    discrete_treatment=True,
    discrete_outcome=True,
    random_state=42
)
cf_01.fit(Y_train[mask_01], T_train[mask_01], X=X_train[mask_01])

cf01_res = build_cf_result_lite(
    cf_model=cf_01,
    X_test=X_test,
    T0=0,
    T1=1,
    Y_test=Y_test,
    T_test=T_test,
    tag="T0_vs_T1"
)

save_cf_result_lite(cf01_res, "./results/cf_T01_lite.pkl")
plot_ite_histogram(cf01_res, "./results/ite_T01.png")
print_ite_analysis(cf01_res)

# ==============================================
#               Compare T0 and T2
# ==============================================
mask_02 = (T_train == 0) | (T_train == 2)
cf_02 = CausalForestDML(
    model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
    model_y=RandomForestClassifier(n_estimators=100, max_depth=5),
    n_estimators=100,
    min_samples_leaf=20,
    discrete_treatment=True,
    discrete_outcome=True,
    random_state=42
)
cf_02.fit(Y_train[mask_02], T_train[mask_02], X=X_train[mask_02])

cf02_res = build_cf_result_lite(
    cf_model=cf_02,
    X_test=X_test,
    T0=0,
    T1=2,
    Y_test=Y_test,
    T_test=T_test,
    tag="T0_vs_T2"
)

save_cf_result_lite(cf02_res, "./results/cf_T02_lite.pkl")
plot_ite_histogram(cf02_res, "./results/ite_T02.png")
print_ite_analysis(cf02_res)








