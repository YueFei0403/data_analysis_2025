# cf_helpers.py

from dataclasses import dataclass
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


@dataclass
class CFResultLite:
    tag: str
    ITE: np.ndarray
    ITE_lower: np.ndarray
    ITE_upper: np.ndarray
    ATE: float
    StdDev: float
    Min: float
    Max: float
    X_test: np.ndarray
    Y_test: np.ndarray
    T_test: np.ndarray


def build_cf_result_lite(cf_model, X_test, T0, T1, Y_test=None, T_test=None, tag="T0_vs_T1"):
    te = cf_model.effect(X_test, T0=T0, T1=T1)
    te_lower, te_upper = cf_model.effect_interval(X_test, T0=T0, T1=T1, alpha=0.05)

    return CFResultLite(
        tag=tag,
        ITE=te,
        ITE_lower=te_lower,
        ITE_upper=te_upper,
        ATE=float(te.mean()),
        StdDev=float(te.std()),
        Min=float(te.min()),
        Max=float(te.max()),
        X_test=X_test,
        Y_test=Y_test,
        T_test=T_test,
    )


def save_cf_result_lite(result, filename):
    with open(filename, "wb") as f:
        pickle.dump(result, f)


def load_cf_result_lite(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def _add_hist_percentage_labels(counts, bins, patches):
    total = np.sum(counts)
    for count, left, patch in zip(counts, bins[:-1], patches):
        if count == 0:
            continue
        pct = 100 * count / total
        plt.text(
            left + (bins[1] - bins[0]) / 2,
            patch.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=45
        )


def plot_ite_histogram(result, filename=None, bins=50):
    te = result.ITE
    counts, bin_edges, patches = plt.hist(
        te,
        bins=bins,
        alpha=0.7,
        color="blue",
        edgecolor="black"
    )

    _add_hist_percentage_labels(counts, bin_edges, patches)

    plt.axvline(result.ATE, color="red", linestyle="--",
                label=f"Mean Effect: {result.ATE:.2f}")

    plt.xlabel(f"ITE ({result.tag})")
    plt.ylabel("Frequency")
    plt.title(f"ITE Distribution — {result.tag}")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)

    plt.close()


# ------------------------------------------------------------
# NEW: Numerical analysis function (matches your formatting)
# ------------------------------------------------------------

def print_ite_analysis(result):
    ite = np.array(result.ITE).flatten()

    print("=================================================")
    print(f"       Numerical Breakdown of ITE {result.tag}       ")
    print("=================================================")

    print(f"Count: {len(ite)}")
    print(f"Mean (ATE): {ite.mean():.4f}")
    print(f"Std Dev: {ite.std():.4f}")
    print(f"Min: {ite.min():.4f}")
    print(f"Max: {ite.max():.4f}")

    # Percentiles
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(ite, p):.4f}")

    # Skewness / Kurtosis
    print(f"Skewness: {skew(ite):.4f}")
    print(f"Kurtosis: {kurtosis(ite):.4f}")  # excess kurtosis

    # Positive/Negative breakdown
    pos_count = np.sum(ite > 0)
    neg_count = np.sum(ite < 0)
    zero_count = np.sum(np.isclose(ite, 0))

    print("----------------------------------------")
    print(f"Positive Effects: {pos_count} ({pos_count/len(ite):.2%})")
    print(f"Negative Effects: {neg_count} ({neg_count/len(ite):.2%})")
    print(f"Zero (≈ no effect): {zero_count} ({zero_count/len(ite):.2%})")
    print("=================================================")
