import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
import os
from datetime import datetime

# --- Logging setup ---
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logfile = os.path.join(log_dir, f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = TeeLogger(logfile)
print(f"📜 Logging all output to {logfile}")


# ==============================
# 辅助函数
# ==============================
def check_users(df, stage, users_before=None):
    """打印当前阶段数据规模与用户变化"""
    n_rows = len(df)
    n_users = df['userid'].nunique()
    print(f"🧾 {stage}: {n_rows} 行, {n_users} 唯一用户")
    if users_before is not None:
        lost = users_before - set(df['userid'])
        print(f"   ↳ 本阶段丢失 {len(lost)} 个用户")
        if len(lost) > 0:
            pd.DataFrame({'lost_userid': list(lost)}).to_csv(f'./lost_users_{stage}.csv', index=False)
            print(f"   📁 已导出: ./lost_users_{stage}.csv")
    return set(df['userid'])

# ==============================
# 1. 读取数据与列名标准化
# ==============================
df = pd.read_csv('./user_data.csv')
df.columns = [c.strip().lower() for c in df.columns]

users_before = check_users(df, "原始数据读取")

# ==============================
# 2. userid 唯一性检查
# ==============================
dup_mask = df.duplicated(subset=['userid'], keep=False)
if dup_mask.any():
    print(f"⚠️ 发现 {dup_mask.sum()} 条重复记录 ({df.loc[dup_mask,'userid'].nunique()} 用户)")
    dup_users = df.loc[dup_mask, 'userid'].unique()
    print("以下为部分重复用户:", dup_users[:10])
    print(df.loc[dup_mask, ['userid', 'group', 'is_add']].sort_values('userid').head(20))
    df.loc[dup_mask].to_csv('./duplicate_users_detail.csv', index=False)
    print("📁 重复用户详情已导出: ./duplicate_users_detail.csv")

    df_no_exact_dups = df.drop_duplicates(keep='first')
    still_dup_mask = df_no_exact_dups.duplicated(subset=['userid'], keep=False)
    conflict_df = df_no_exact_dups[still_dup_mask]
    if not conflict_df.empty:
        print("❌ 同一 userid 存在不同记录（冲突），请人工检查。")
        conflict_df.to_csv('./user_conflicts.csv', index=False)
        raise ValueError("存在冲突用户记录，停止执行。")
    else:
        df = df_no_exact_dups
        print("✅ 仅存在完全重复行，已自动去重。")

users_before = check_users(df, "去重后", users_before)

# ==============================
# 3. 分组唯一性与逻辑检查
# ==============================
group_per_user = df.groupby('userid')['group'].nunique()
bad = group_per_user[group_per_user > 1]
if len(bad) > 0:
    bad_users = bad.index.tolist()
    print(f"❌ 有 {len(bad_users)} 个用户出现在多个分组: {bad_users[:10]}")
    pd.DataFrame({'bad_userid': bad_users}).to_csv('./conflict_groups.csv', index=False)
    raise ValueError("实验随机化被破坏：同一用户出现在多个组。")

print("✅ 用户分组一致性通过。")

# ==============================
# 4. 数值与逻辑合法性
# ==============================
num_cols = ['90days_purchase_time','90days_per_purchase_price','90days_purchase_amount',
            '90days_coupon_time','90days_coupon_ratio','last_purchase_day']

for c in num_cols:
    if c in df.columns:
        if (df[c] < 0).any():
            raise ValueError(f"{c} 存在负值。")

mask_invalid = df['90days_coupon_time'] > df['90days_purchase_time']
if mask_invalid.any():
    print(f"⚠️ {mask_invalid.sum()} 条优惠券次数>购买次数，已标记。")
    df['data_flag'] = np.where(mask_invalid, 'coupon>purchase', 'ok')
else:
    df['data_flag'] = 'ok'

# ==============================
# 5. 缺失值与衍生特征
# ==============================
for col in ['90days_purchase_time', '90days_per_purchase_price',
            '90days_purchase_amount', '90days_coupon_time', 'last_purchase_day']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

df['90days_coupon_ratio'] = (
    df['90days_coupon_time'] / df['90days_purchase_time'].replace(0, np.nan)
).fillna(0)

df['coupon_per_purchase'] = df['90days_coupon_time'] / df['90days_purchase_time'].replace(0, 1)
df['activity_score'] = 1 / (df['last_purchase_day'] + 1)
df['price_sensitivity'] = df['90days_coupon_ratio'] * df['90days_per_purchase_price']

users_before = check_users(df, "特征衍生后", users_before)

# ==============================
# 6. Group 编码 + OS 独热
# ==============================
df['group'] = df['group'].map({
    'control': 0, '对照组': 0,
    'exp1': 1, '实验组1': 1,
    'exp2': 2, '实验组2': 2
})
if 'user_os' in df.columns:
    df = pd.get_dummies(df, columns=['user_os'], drop_first=True)

# ==============================
# 7. 异常值检测（仅标记不删除）
# ==============================
for col in ['90days_purchase_time','90days_purchase_amount','90days_coupon_time']:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper, lower = q3 + 3*iqr, q1 - 3*iqr
    outliers = df[(df[col] > upper) | (df[col] < lower)]
    if len(outliers) > 0:
        print(f"⚠️ {col} 检测到 {len(outliers)} 条异常值 (未删除，仅标记)")
        df.loc[outliers.index, 'data_flag'] = 'outlier'

users_before = check_users(df, "异常值检测后", users_before)

# ==============================
# 8. 数值标准化 + 分组平衡检验
# ==============================
scaler = StandardScaler()
num_cols_all = ['90days_purchase_time','90days_per_purchase_price','90days_purchase_amount',
                '90days_coupon_time','90days_coupon_ratio','last_purchase_day',
                'coupon_per_purchase','activity_score','price_sensitivity']
df[num_cols_all] = scaler.fit_transform(df[num_cols_all])

report = []
for col in num_cols_all:
    stat, p = stats.kruskal(df[df['group']==0][col],
                            df[df['group']==1][col],
                            df[df['group']==2][col])
    report.append({'feature': col, 'p_value': p})
report_df = pd.DataFrame(report)
report_df['significant_diff'] = report_df['p_value'] < 0.05

# ==============================
# 9. 输出结果
# ==============================
df.to_csv('./ab_coupon_clean_debug.csv', index=False)
report_df.to_csv('./group_balance_report.csv', index=False)

print("\n✅ 数据清洗完成")
print("📁 清洗后: ./ab_coupon_clean_debug.csv")
print("📁 分组平衡报告: ./group_balance_report.csv")
