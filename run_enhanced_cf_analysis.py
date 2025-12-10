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

# Create global timestamped directory
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

# Redirect stdout and stderr to a fresh file
sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout  # optional — log errors too
print(f"[INFO] Logging initialized → {log_path}")

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
        ("num", "passthrough", numeric_features), # <--- no scaling
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

def extract_feature_matrices(df):
    X_df = df[FEATURE_COLS].copy()
    Y_arr = df[OUTCOME_COL].astype(int).values.ravel()
    
    df["group_code"] = df[TREATMENT_COL].astype("category").cat.codes
    T_arr = df["group_code"].values
    
    X_arr = preprocessor.fit_transform(X_df)
    
    return X_arr, Y_arr, T_arr, preprocessor


X_arr, Y_arr, T_arr, preprocessor = extract_feature_matrices(df)
print("X shape:", X_arr.shape)
print("Y shape:", Y_arr.shape)
print("T shape:", T_arr.shape)

# Dataset A: T1 vs Control
mask_T1 = (T_arr == 0) | (T_arr == 1)
X_T1 = X_arr[mask_T1]
Y_T1 = Y_arr[mask_T1]
T_T1 = (T_arr[mask_T1] == 1).astype(int)   # 1 if T1, else 0

# Dataset B: T2 vs Control
mask_T2 = (T_arr == 0) | (T_arr == 2)
X_T2 = X_arr[mask_T2]
Y_T2 = Y_arr[mask_T2]
T_T2 = (T_arr[mask_T2] == 2).astype(int)   # 1 if T2, else 0

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

def choose_strategy(tau1, tau2):
    """
    Decision rule:
    - If both ≤ 0 → stay in control group
    - Otherwise choose argmax
    """
    if tau1 <= 0 and tau2 <= 0:
        return 0  # control
    if tau1 > tau2:
        return 1  # T1
    else:
        return 2  # T2
    
df["best_strategy"] = [
    choose_strategy(t1, t2) for t1, t2 in zip(df["cate_T1"], df["cate_T2"])
]

# Map back to readable labels
df["best_strategy_label"] = df["best_strategy"].map({
    0: "对照组",
    1: "实验组1",
    2: "实验组2"
})

########################################################
#                    Step 1 -- CATE                    #
########################################################
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

def compute_qini_curve(ite, y, t_bin):
    ite = np.asarray(ite).flatten()
    y = np.asarray(y).flatten()
    t_bin = np.asarray(t_bin).flatten()

    # Sort by uplift descending
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

    # Avoid division by zero
    treat_rate = df_eval['y_treat_cum'] / df_eval['n_treat_cum'].replace(0, np.nan)
    control_rate = df_eval['y_control_cum'] / df_eval['n_control_cum'].replace(0, np.nan)

    # Incremental gain curve
    df_eval['uplift_gain'] = (treat_rate - control_rate) * df_eval['n_treat_cum']
    df_eval['uplift_gain'] = df_eval['uplift_gain'].fillna(0)

    # X-axis: 0→1
    x_axis = np.arange(N) / N

    return x_axis, df_eval['uplift_gain'].values

########################################################
#                    Step 2 -- Best Strategy           #
########################################################
def print_tree_with_gini_best_confidence(clf, feature_names, class_names, node_id=0, indent=""):
    tree = clf.tree_

    gini = tree.impurity[node_id]
    samples = tree.n_node_samples[node_id]
    values = tree.value[node_id][0]  # class counts per node

    best_strategy = values.argmax()
    confidence = values.max() / values.sum() if values.sum() > 0 else 0.0

    # Leaf node
    if tree.children_left[node_id] == -1:
        print(
            f"{indent}Leaf: Gini={gini:.3f}, Samples={samples}, "
            f"ClassDist={values.tolist()}, "
            f"BestStrategy={best_strategy}, Confidence={confidence:.3f}"
        )
        return

    # Split rule
    feature = feature_names[tree.feature[node_id]]
    threshold = tree.threshold[node_id]

    print(
        f"{indent}Node: if {feature} <= {threshold:.3f} "
        f"(Gini={gini:.3f}, Samples={samples}, "
        f"ClassDist={values.tolist()}, "
        f"BestStrategy={best_strategy}, Confidence={confidence:.3f})"
    )

    # Left branch
    print(f"{indent}├── True branch:")
    print_tree_with_gini_best_confidence(
        clf, feature_names, class_names, tree.children_left[node_id], indent + "│   "
    )

    # Right branch
    print(f"{indent}└── False branch:")
    print_tree_with_gini_best_confidence(
        clf, feature_names, class_names, tree.children_right[node_id], indent + "    "
    )
    

def build_labels(clf, class_names):
    tree = clf.tree_
    labels = []

    for node_id in range(tree.node_count):
        values = tree.value[node_id][0]
        samples = tree.n_node_samples[node_id]

        best_strategy = values.argmax()
        confidence = values.max() / values.sum() if values.sum() > 0 else 0

        label = (
            f"Best: {class_names[best_strategy]}\n"
            f"Conf: {confidence:.2f}\n"
            f"Samples: {samples}"
        )
        labels.append(label)
    return labels

def plot_strategy_tree(clf, feature_names, class_names, save_path=None):
    labels = build_labels(clf, class_names)

    plt.figure(figsize=(40, 12))

    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        impurity=False,       # remove sklearn impurity
        label="none",         # remove sklearn labels
        filled=True,
        rounded=True,
        fontsize=11,
        node_ids=True         # helps map node_id to label
    )

    # Replace default labels with minimal labels
    ax = plt.gca()
    for text_obj in ax.get_children():
        if isinstance(text_obj, plt.Text):
            try:
                node_id = int(text_obj.get_text().split()[0])  # first number = node idx
                text_obj.set_text(labels[node_id])
            except:
                continue

    if save_path:
        plt.savefig(save_path, dpi=240, bbox_inches="tight")
        print(f"[Saved strategy tree → {save_path}]")

    plt.title("Best Strategy Decision Tree", fontsize=16)
    

X_features = X_arr.copy()        # original features (before encoding)
y_strategy = df["best_strategy"] # 0,1,2

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

print("\n========= Best Strategy =========")
print_tree_with_gini_best_confidence(clf, feature_names, class_names)


# Summarize tree leaves -- Automated
def summarize_decision_tree_leaves(clf, feature_names, class_names):
    """
    Iteratively traverse a DecisionTreeClassifier and summarize each leaf node.
    """

    tree = clf.tree_
    leaf_summaries = []

    # Stack holds tuples: (node_id, path_rules)
    stack = [(0, [])]  # start at root node

    while stack:
        node_id, path_rules = stack.pop()
        
        # Leaf node
        if tree.children_left[node_id] == -1:
            values = tree.value[node_id][0]
            samples = tree.n_node_samples[node_id]

            best_strategy_idx = int(values.argmax())
            best_strategy = class_names[best_strategy_idx]
            confidence = values.max() / values.sum() if values.sum() > 0 else 0.0

            leaf_summaries.append({
                "leaf_id": node_id,
                "rule": path_rules,
                "best_strategy": best_strategy,
                "best_strategy_index": best_strategy_idx,
                "confidence": confidence,
                "samples": samples,
                "class_distribution": values.tolist(),
                "insight": (
                    f"Segment of {samples} users. Best strategy = {best_strategy} "
                    f"(confidence={confidence:.2f})."
                )
            })
            continue

        # Internal node → get split feature + threshold
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]

        # LEFT child → condition true
        left_rule = f"{feature} <= {threshold:.3f}"
        stack.append((tree.children_left[node_id], path_rules + [left_rule]))

        # RIGHT child → condition false
        right_rule = f"{feature} > {threshold:.3f}"
        stack.append((tree.children_right[node_id], path_rules + [right_rule]))

    return leaf_summaries


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
#       Step 3 -- Compute Qini for each treatment      #
########################################################
mask_T1 = (T_arr == 0) | (T_arr == 1)
mask_T2 = (T_arr == 0) | (T_arr == 2)

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
    
    
