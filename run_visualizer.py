import pandas as pd
import numpy as np

def generate_strategy_summary_table(df, numeric_features, categorical_feature):
    strategy_map = {0: "Control", 1: "T1", 2: "T2"}
    df["strategy"] = df["best_strategy"].map(strategy_map)

    strategies = ["Control", "T1", "T2"]

    # -------- 数值特征表 --------
    table_numeric = []

    for feat in numeric_features:
        row = [feat]
        for strat in strategies:
            vals = df.loc[df["strategy"] == strat, feat]
            mean = vals.mean()
            std = vals.std()
            row.append(f"{mean:.2f} ± {std:.2f}")
        table_numeric.append(row)

    numeric_df = pd.DataFrame(
        table_numeric,
        columns=["Feature", "Control", "T1", "T2"]
    )

    # -------- 分类特征表 --------
    cat_counts = df.groupby(["strategy", categorical_feature]).size().unstack(fill_value=0)
    cat_pct = cat_counts.div(cat_counts.sum(axis=1), axis=0) * 100
    cat_pct = cat_pct.loc[strategies]  # enforce order

    return numeric_df, cat_pct



def print_group_stat(df):
    """
    Print strategy target summary in Markdown table format.
    """
    strategy_map = {0: "Control", 1: "T1", 2: "T2"}
    df["strategy"] = df["best_strategy"].map(strategy_map)

    # Count number per strategy
    strat_counts = df["strategy"].value_counts().reindex(["Control", "T1", "T2"], fill_value=0)
    total = strat_counts.sum()

    # Prepare table-like output
    print("\n=== Final Strategy Target Summary ===")
    print(f"{'Strategy':<12} {'Users':>10} {'Percent':>12}")
    print("-" * 36)
    print("\n### Final Strategy Target Summary\n")

    # Markdown header
    print("| Strategy | Users | Percent |")
    print("|----------|-------|---------|")

    # Rows
    for strat, count in strat_counts.items():
        pct = count / total * 100
        print(f"| {strat} | {count} | {pct:.2f}% |")

    # Total row
    print(f"| **Total** | **{total}** | **100.00%** |")
