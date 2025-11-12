"""
双因子因果分
- 主效应A：Popup vs Banner
- 主效应B：Coupon vs Non-Coupon（在弹窗内：组1 vs 组2）
- 交互项：Popup × Coupon（OLS 回归）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.stats import ttest_1samp
import statsmodels.formula.api as smf


# === setups ===
INPUT_CSV = "./ab_coupon_clean_debug.csv"
OUTPUT_DIR = "./results"
RANDOM_STATE = 42

# === Read in and Basic Checkups ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_CSV)

required_cols = {
    "userid", "user_os_iOS", "90days_purchase_time", "90days_per_purchase_price",
    "90days_purchase_amount", "90days_coupon_time", "90days_coupon_ratio",
    "last_purchase_day", "group", "is_add"
}
missing = required_cols - set(df.columns)
assert not missing, f"Missing necessary columns: {missing}"

# Derived Dual Factors
df["A_popup"] = (df["group"] != 0).astype(int)  # whether it is popup or not
df["B_coupon"] = (df["group"] == 2).astype(int) # whether it contains $10 coupon
df["is_add"] = df["is_add"].astype(int)
df["user_os_iOS"] = df["user_os_iOS"].astype(int)

# Features: numerical + categorical
feature_cols = [
    "90days_purchase_time", "90days_per_purchase_price", "90days_purchase_amount",
    "90days_coupon_time", "90days_coupon_ratio", "last_purchase_day",
    "coupon_per_purchase", "activity_score", "price_sensitivity", "user_os_iOS"
]
X = df[feature_cols].values
Y = df["is_add"].astype(int).values

# ========== 1. Popup main effect ==========
print("\n>>> Estimating Popup (A) effect >>>")
df_A = df[df["group"].isin([0, 1, 2])].copy()
X_A = df_A[feature_cols].values
Y_A = df_A["is_add"].astype(int).values
T_A = df_A["A_popup"].fillna(0).astype(int).values

cf_A = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE),
    model_t=RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE),
    discrete_outcome=True,
    discrete_treatment=True,
    n_estimators=1000, random_state=RANDOM_STATE
)
cf_A.fit(Y_A, T_A, X=X_A)
tau_A = cf_A.effect(X_A)
df_A["tau_A"] = tau_A
ate_A = cf_A.ate(X_A)
p_A = ttest_1samp(tau_A, 0).pvalue

ate_A = float(ate_A)
p_A = float(p_A)
print(f"ATE(Popup vs Banner) = {ate_A:.4f}, p = {p_A:.3e}")

# Save and plot
df_A.to_csv(os.path.join(OUTPUT_DIR, "popup_effect.csv"), index=False)
plt.figure(figsize=(8,4))
plt.hist(tau_A, bins=30)
plt.title("CATE Distribution: Popup vs Banner")
plt.xlabel("Estimated Effect (τ_A)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cate_popup.png"), dpi=300)
plt.close()
print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

# ========== 2️. Coupon main effect ==========
print("\n>>> Estimating Coupon (B) effect >>>")
df_B = df[df["group"].isin([1, 2])].copy()
X_B = df_B[feature_cols].values
Y_B = df_B["is_add"].astype(int).values
T_B = df_B["B_coupon"].astype(int).values

cf_B = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE),
    model_t=RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE),
    discrete_outcome=True,
    discrete_treatment=True,
    n_estimators=1000, random_state=RANDOM_STATE
)
cf_B.fit(Y_B, T_B, X=X_B)
tau_B = cf_B.effect(X_B)
df_B["tau_B"] = tau_B
ate_B = cf_B.ate(X_B)
p_B = ttest_1samp(tau_B, 0).pvalue

ate_B = float(ate_B)
p_B = float(p_B)
print(f"ATE(Coupon vs Non-Coupon) = {ate_B:.4f}, p = {p_B:.3e}")

# Save and plot
df_B.to_csv(os.path.join(OUTPUT_DIR, "coupon_effect.csv"), index=False)
plt.figure(figsize=(8,4))
plt.hist(tau_B, bins=30)
plt.title("CATE Distribution: Coupon vs Non-Coupon")
plt.xlabel("Estimated Effect (τ_B)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cate_coupon.png"), dpi=300)
plt.close()
print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")


# ========== 3. Investigate the interaction of popup and coupon ==========