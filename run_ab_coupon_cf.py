import pandas as pd
from io import StringIO
from econml.dml import CausalForestDML
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
# === Step 1 ===
df = pd.read_csv("user_data.csv")
df.describe()
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

print("Shapes before split:")
print("X:", df_X.shape)
print("Y:", df_Y.shape)
print("T:", np.shape(T_arr))
print("W:", W_arr.shape)
X_train, X_test, Y_train, Y_test, T_train, T_test, W_train, W_test = train_test_split(
    X_arr, Y_arr, T_arr, W_arr, test_size=0.3, random_state=42
)

cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
    model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
    n_estimators=100,
    min_samples_leaf=20,
    discrete_treatment=True,
    random_state=42
)
cf_model.fit(Y_train, T_train, X=X_train, W=W_train)

te_pred = cf_model.effect(X_test)
te_lower, te_upper = cf_model.effect_interval(X_test, alpha=0.5)

print(f"平均处理效应（ATE）: {te_pred.mean():.2f}")
print(f"处理效应标准差: {te_pred.std():.2f}")
print(f"最小处理效应: {te_pred.min():.2f}")
print(f"最大处理效应: {te_pred.max():.2f}")


plt.figure(figsize=(12,5))

plt.hist(te_pred, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(te_pred.mean(), color='red', linestyle='--', label=f'Mean Effect: {te_pred.mean():.2f}')
plt.xlabel('Treatment Effect')
plt.ylabel('Frequency')
plt.title('Individual Treatment Effect (ITE) Distribution')
plt.legend()