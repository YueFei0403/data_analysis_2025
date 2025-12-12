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

from helpers import *

from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt

from datetime import datetime

############################################
#           Create Output Directory        #
############################################
timestamp = datetime.now().strftime("%Y%m%d")
OUTPUT_DIR = os.path.join("results", timestamp)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] All outputs will be saved to: {OUTPUT_DIR}")

log_path = os.path.join(OUTPUT_DIR, "output.log")
if os.path.exists(log_path):
    os.remove(log_path)
    print(f"[INFO] Old log file removed: {log_path}")
else:
    print(f"[INFO] No previous log file found at: {log_path}")

sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout # optional -> log errors too
print(f"[INFO] Logging initialized -> {log_path}")

############################################
#              Data Reading                #
############################################
df = pd.read_csv("user_data.csv")

FEATURE_COLS = [
    "User_os",
    "90days_purchase_time",
    "90days_per_purchase_price",
    "90days_purchase_amount",
    "90days_coupon_time",
    "90days_coupon_ratio",
    "Last_purchase_day",
]

OUTCOME_COL = "is_add"
TREATMENT_COL = "group"

numeric_features = [
    "90days_purchase_time",
    "90days_per_purchase_price",
    "90days_purchase_amount",
    "90days_coupon_time",
    "90days_coupon_ratio",
    "Last_purchase_day",
]

categorical_features = ["User_os"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features), # <-- no scaling
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

############################################
#               Modeling                   #
############################################
X_arr, Y_arr, T_arr, preprocessor = extract_feature_matrices(df, preprocessor)
print("X shape:", X_arr.shape)
print("Y shape:", Y_arr.shape)
print("T shape:", T_arr.shape)

# Dataset A: T1 vs Control
mask_T1 = (T_arr == 0) | (T_arr == 1)
X_T1 = X_arr[mask_T1]
Y_T1 = Y_arr[mask_T1]
T_T1 = (T_arr[mask_T1] == 1).astype(int)

# Dataset B: T2 vs Control
mask_T2 = (T_arr == 0) | (T_arr == 2)
X_T2 = X_arr[mask_T2]
Y_T2 = Y_arr[mask_T2]
T_T2 = (T_arr[mask_T2] == 2).astype(int)

cf_01 = CausalForestDML(
    model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
    model_y=RandomForestClassifier(n_estimators=100, max_depth=5),
    n_estimators=100,
    min_samples_leaf=20,
    discrete_treatment=True,
    discrete_outcome=True,
    random_state=42
)

cf_02 = CausalForestDML(
    model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
    model_y=RandomForestClassifier(n_estimators=100, max_depth=5),
    n_estimators=100,
    min_samples_leaf=20,
    discrete_treatment=True,
    discrete_outcome=True,
    random_state=42
)

print("\nTraining model for T1 vs Control...")
cf_01.fit(Y_T1, T_T1, X=X_T1)

print("\nTraining model for T2 vs Control...")
cf_02.fit(Y_T2, T_T2, X=X_T2)

print("\nInferences...")
cate_T1 = cf_01.effect(X_arr)   # effect of T1 vs Control
cate_T2 = cf_02.effect(X_arr)   # effect of T2 vs Control

df["cate_T1"] = cate_T1
df["cate_T2"] = cate_T2

#####################################################
#     Classification - Choose the Best Strategy     #
#####################################################
df["best_strategy"] = [
    choose_strategy(t1, t2) for t1, t2 in zip(df["cate_T1"], df["cate_T2"])
]

df["best_strategy_label"] = df["best_strategy"].map(
    {
        0: "Control",
        1: "T1",
        2: "T2"
    }
)

#####################################################
#                   Plot CATE Values                #
#####################################################
print("\n=== Final Output ===")
print(df[["Userid", "cate_T1", "cate_T2", "best_strategy_label"]])

plt.figure(figsize=(10, 5))
plt.hist(df["cate_T1"], bins=40, alpha=0.6, label="CATE T1", color="blue")
plt.hist(df["cate_T2"], bins=40, alpha=0.6, label="CATE T2", color="green")
plt.title("CATE Distribution for T1 and T2")
plt.xlabel("CATE Value")
plt.ylabel("User Count")
plt.legend()
plt.grid(alpha=0.3)
filename = OUTPUT_DIR + "/cate_overall.png"
if filename:
    plt.savefig(filename, dpi=500)
    
    
X_features = X_arr.copy()
y_strategy = df["best_strategy"]

clf = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=50
)
clf.fit(X_features, y_strategy)

feature_names = preprocessor.get_feature_names_out(FEATURE_COLS)
class_names = ["Control", "T1", "T2"]

plot_strategy_tree(
    clf,
    feature_names,
    class_names,
    save_path=os.path.join(OUTPUT_DIR, "strategy_tree.png")
)

print("\n=========== BEST STRATEGY ===========")
print_tree_with_gini_best_confidence(clf, feature_names, class_names)

leaf_summaries = summarize_decision_tree_leaves(
    clf, feature_names, class_names
)
for leaf in leaf_summaries:
    print("\n=== Leaf", leaf["leaf_id"], "===")
    print("Rules:")
    for r in leaf["rule"]:
        print("  -", r)
    print("Best Strategy:", leaf["best_strategy"])
    print("Confidence:", f"{leaf['confidence']:.2f}")
    print("Samples:", leaf["samples"])
    print("Class Dist:", leaf["class_distribution"])
    print("Insight:", leaf["insight"])



########################################################
#       Re-evaluate Model Segmentation Ability         #
########################################################   
# T1 vs Control
x1, qini1 = compute_qini_curve(
    ite=cate_T1[mask_T1],
    y=Y_arr[mask_T1],
    t_bin=(T_arr[mask_T1] == 1).astype(int)
)

# T2 vs Control
x2, qini2 = compute_qini_curve(
    ite=cate_T2[mask_T2],
    y=Y_arr[mask_T2],
    t_bin=(T_arr[mask_T2] == 2).astype(int)
)

plt.figure(figsize=(10, 6))
plt.plot(x1, qini1, label="Qini Curve T1", linewidth=2)
plt.plot(x2, qini2, label="Qini Curve T2", linewidth=2)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Qini Curves for T1 and T2")
plt.xlabel("Population Portion (sorted by uplift)")
plt.ylabel("Cumulative Incremental Gain")
plt.grid(alpha=0.3)
plt.legend()
filename = OUTPUT_DIR + "/qini_overall.png"
if filename:
    plt.savefig(filename, dpi=300)
