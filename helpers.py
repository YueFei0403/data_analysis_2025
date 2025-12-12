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

def extract_feature_matrices(df, preprocessor):
    X_df = df[FEATURE_COLS].copy()
    Y_arr = df[OUTCOME_COL].astype(int).values.ravel()

    df["group_code"] = df[TREATMENT_COL].astype("category").cat.codes
    T_arr = df["group_code"].values
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
    
    # Avoid division by zero
    treat_rate = df_eval['y_treat_cum'] / df_eval['n_treat_cum'].replace(0, np.nan)
    control_rate = df_eval['y_control_cum'] / df_eval['n_control_cum'].replace(0, np.nan)
    
    # Incremental gain curve
    df_eval['uplift_gain'] = (treat_rate - control_rate) * df_eval['n_treat_cum']
    df_eval['uplift_gain'] = df_eval['uplift_gain'].fillna(0)
    
    # X-axis: 0 -> 1
    x_axis = np.arange(N) / N
    
    return x_axis, df_eval['uplift_gain'].values


"""
    At Node t, it contains N_{t,c} samples belonging to class c given that
    it contains N_t samples:
        Gini(t) = 1 - sum_{c=1}^{C} p(c|t)^2
"""
def print_tree_with_gini_best_confidence(clf, feature_names, class_names, node_id=0, indent=""):
    tree = clf.tree_
    
    gini = tree.impurity[node_id]
    samples = tree.n_node_samples[node_id]
    values = tree.value[node_id][0]
    
    best_strategy = values.argmax()
    best_strategy_gain = values.max()
    overall_gain = values.sum()
    confidence = best_strategy_gain / overall_gain if overall_gain > 0 else 0.0
    
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
        best_strategy_gain = values.max()
        overall_gain = values.sum()
        confidence = best_strategy_gain / overall_gain if overall_gain > 0 else 0.0
        
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