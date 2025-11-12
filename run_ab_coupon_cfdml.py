import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
# === Step 1 ===
df = pd.read_csv("user_data.csv")
print(df.describe())
# === Step 2 ===
# Tag
C = ["Userid"]

# 混淆变量
W = [
    "User_os",
    "90days_purchase_time",
    "90days_per_purchase_price",
    "90days_purchase_amount",
    "90days_coupon_time",
    "Last_purchase_day"
]

# 特征变量
X = ["90days_coupon_ratio"]

# 处理变量 (Treatment)
T = ["group"]

# 结果变量 （Outcome)
Y = ["is_add"]

# === Step 3 ===
df_X = df[X]
df_Y = df[Y]
df_T = df[T]
df_W = df[W]

# === Step 4 ===
# User_os is categorical, others numeric
preprocessor = ColumnTransformer([
    ('os', OneHotEncoder(drop='first'), ['User_os']),
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

W_proc = preprocessor.fit_transform(df_W)

# === Step 5 ===
X_arr = df_X.values            
Y_arr = df_Y.values.ravel()    
T_arr = df["group_code"].values                   
W_arr = W_proc                 

print("\n>>> Shapes before split: >>>")
print("X:", df_X.shape)
print("Y:", df_Y.shape)
print("T:", np.shape(T_arr))
print("W:", W_arr.shape)
X_train, X_test, Y_train, Y_test, T_train, T_test, W_train, W_test = train_test_split(
    X_arr, Y_arr, T_arr, W_arr, test_size=0.3, random_state=42
)

# To better understand how balanced my experimental/causal-inference setup is:
print("\n>>> Checking Treatment Class Balance >>>")
unique, counts = np.unique(T_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Treatment {u}: {c} samples ({c / len(T_train):.2%})")

# === Compare T0 and T1 ===
mask_01 = (T_train == 0) | (T_train == 1)
cf_01 = CausalForestDML(
    model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
    model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
    n_estimators=100,
    min_samples_leaf=20,
    discrete_treatment=True,
    discrete_outcome=True,
    random_state=42
)
te_01_pred = cf_01.effect(X_test)
te_01_lower, te_01_upper = cf_01.effect_interval(X_test, alpha=0.05)
print(f"Average Treatment Effect (ATE): {te_01_pred.mean():.2f}")
print(f"Treatment Effect Std-Dev: {te_01_pred.std():.2f}")
print(f"Treatment Effect Min: {te_01_pred.min():.2f}")
print(f"Treatment Effect Max: {te_01_pred.max():.2f}")
cf_01.fit(Y_train[mask_01], T_train[mask_01], X=X_train[mask_01])
print("\n <<<<<<<< CausalForestDML (T0 vs T1) Training Completed !!! <<<<<<<<<<<< \n")

# === Compare T0 and T2 ===
mask_02 = (T_train == 0) | (T_train == 2)
cf_02 = CausalForestDML(
    model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
    model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
    n_estimators=100,
    min_samples_leaf=20,
    discrete_treatment=True,
    discrete_outcome=True,
    random_state=42
)
cf_02.fit(Y_train[mask_02], T_train[mask_02], X=X_train[mask_02])
print("\n <<<<<<<< CausalForestDML (T0 vs T2) Training Completed !!! <<<<<<<<<<<< \n")
te_02_pred = cf_02.effect(X_test)
te_02_lower, te_02_upper = cf_02.effect_interval(X_test, alpha=0.05)
print(f"Average Treatment Effect (ATE): {te_02_pred.mean():.2f}")
print(f"Treatment Effect Std-Dev: {te_02_pred.std():.2f}")
print(f"Treatment Effect Min: {te_02_pred.min():.2f}")
print(f"Treatment Effect Max: {te_02_pred.max():.2f}")