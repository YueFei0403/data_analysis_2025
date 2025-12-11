import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
import xgboost as xgb

from matplotlib import font_manager

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False   # show minus sign correctly
# ------------------------------------------------------------
# Load stored CFResultLite data
# ------------------------------------------------------------
def load_result(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
# ------------------------------------------------------------
# Extract binary treated/control pair for a specific arm pair
# ------------------------------------------------------------
def extract_pair_data(result, treat_label, control_label):
    ite = np.array(result.ITE).flatten()
    y = np.array(result.Y_test).flatten()
    t = np.array(result.T_test).flatten()

    mask = (t == treat_label) | (t == control_label)

    ite_pair = ite[mask]
    y_pair = y[mask]
    t_pair = t[mask]

    # binary treatment vector: 1 = treat_label, 0 = control_label
    t_bin = (t_pair == treat_label).astype(int)

    return ite_pair, y_pair, t_bin


def compute_qini_curve(ite, y, t_bin):
    ite = np.asarray(ite).flatten()
    y = np.asarray(y).flatten()
    t_bin = np.asarray(t_bin).flatten()

    df_eval = pd.DataFrame({
        'y': y,
        'T': t_bin,
        'uplift': ite
    }).sort_values('uplift', ascending=False)

    N = len(df_eval)
    n_treat_total = df_eval['T'].sum()
    n_control_total = N - n_treat_total
    
    df_eval['n_treat_cum'] = df_eval['T'].cumsum()
    df_eval['n_control_cum'] = (1 - df_eval['T']).cumsum()
    df_eval['y_treat_cum'] = (df_eval['y'] * df_eval['T']).cumsum()
    df_eval['y_control_cum'] = (df_eval['y'] * (1 - df_eval['T'])).cumsum()
    
    df_eval['uplift_gain'] = (df_eval['y_treat_cum'] / df_eval['n_treat_cum']) - (df_eval['y_control_cum'] / df_eval['n_control_cum'])
                                
    df_eval['uplift_gain'] = df_eval['uplift_gain'] * df_eval['n_treat_cum']
    
    df_eval = df_eval.fillna(0)
    
    qini_score = np.trapz(df_eval['uplift_gain'], dx=1/N)
    
    return df_eval, qini_score
    
    
def plot_uplift_curve(df_eval, tag, filename=None):
    N = len(df_eval)
    
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(N) / N, df_eval['uplift_gain'], label=f'模型（{tag}）')
    plt.plot([0, 1], [0, df_eval['uplift_gain'].iloc[-1]], 'k--', label=f'随即策略')
    plt.xlabel("Fraction of population targeted")
    plt.ylabel("Incremental gain (treated vs control)")
    plt.title(f"CausalForestDML 的 Qini曲线 — {tag}")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.close()



# =====================================
#             MAIN Routine
# =====================================
result_01 = load_result("./results/cf_T01_lite.pkl")
result_02 = load_result("./results/cf_T02_lite.pkl")

ite_01, y_01, t_bin_01 = extract_pair_data(result_01, treat_label=1, control_label=0)
ite_02, y_02, t_bin_02 = extract_pair_data(result_02, treat_label=2, control_label=0)

df_eval_01, auuc_01 = compute_qini_curve(ite_01, y_01, t_bin_01)
df_eval_02, auuc_02 = compute_qini_curve(ite_02, y_02, t_bin_02)

plot_uplift_curve(df_eval_01, tag="T0_vs_T1", filename="./results/qini_T01.png")
plot_uplift_curve(df_eval_02, tag="T0_vs_T2", filename="./results/qini_T02.png")

print(f"AUUC (T0 vs T1): {auuc_01:.4f}")
print(f"AUUC (T0 vs T2): {auuc_02:.4f}")
