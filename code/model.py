"""
全国大学生统计建模大赛 - 建模主脚本
模型链：XGBoost变量重要性 → 面板固定效应 → 分位数回归 → 稳健性检验
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('panel_data.csv')

print("=" * 60)
print("第一步：描述性统计")
print("=" * 60)

desc = df[['birth_rate','income','urbanization']].describe().round(2)
print(desc)

# ============================================================
# 图1：出生率时间趋势（分区域）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 全国均值趋势
nat_trend = df.groupby('year')['birth_rate'].mean()
axes[0].plot(nat_trend.index, nat_trend.values, 'b-o', linewidth=2, markersize=5)
axes[0].set_title('全国平均出生率变化趋势（2010-2022）', fontsize=13)
axes[0].set_xlabel('年份')
axes[0].set_ylabel('出生率（‰）')
axes[0].grid(alpha=0.3)
axes[0].axvline(x=2016, color='red', linestyle='--', alpha=0.5, label='全面二孩政策')
axes[0].axvline(x=2021, color='orange', linestyle='--', alpha=0.5, label='三孩政策')
axes[0].legend()

# 分区域趋势
colors = {'东部':'#e74c3c', '中部':'#3498db', '西部':'#2ecc71'}
for region, grp in df.groupby('region'):
    trend = grp.groupby('year')['birth_rate'].mean()
    axes[1].plot(trend.index, trend.values, 'o-', color=colors[region],
                 label=region, linewidth=2, markersize=5)
axes[1].set_title('分区域出生率变化趋势', fontsize=13)
axes[1].set_xlabel('年份')
axes[1].set_ylabel('出生率（‰）')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig1_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ 图1已保存：fig1_trend.png")

# ============================================================
# 第二步：XGBoost 变量重要性
# ============================================================
print("\n" + "=" * 60)
print("第二步：XGBoost 变量重要性分析")
print("=" * 60)

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

df_ml = df.copy()
le = LabelEncoder()
df_ml['province_code'] = le.fit_transform(df_ml['province'])
df_ml['region_code'] = le.fit_transform(df_ml['region'])

features = ['income', 'urbanization', 'log_income', 'year_trend',
            'province_code', 'region_code']
X = df_ml[features]
y = df_ml['birth_rate']

xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                   random_state=42, verbosity=0)
xgb.fit(X, y)

importance = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=True)
readable_names = {
    'income': '人均可支配收入',
    'urbanization': '城镇化率',
    'log_income': '收入对数',
    'year_trend': '时间趋势',
    'province_code': '省份固定效应',
    'region_code': '区域效应'
}
importance.index = [readable_names.get(i, i) for i in importance.index]

fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = ['#e74c3c' if v == importance.max() else '#3498db' for v in importance.values]
bars = ax.barh(importance.index, importance.values, color=colors_bar)
ax.set_title('XGBoost变量重要性排序', fontsize=13)
ax.set_xlabel('重要性得分（F-score）')
for bar, val in zip(bars, importance.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('fig2_xgboost.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图2已保存：fig2_xgboost.png")
print(f"\nXGBoost R²: {xgb.score(X, y):.4f}")

# ============================================================
# 第三步：面板固定效应回归
# ============================================================
print("\n" + "=" * 60)
print("第三步：面板固定效应回归（主回归）")
print("=" * 60)

from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# 设置面板索引
df_panel = df.set_index(['province', 'year'])

# 模型1：基准模型（只有城镇化率）
mod1 = PanelOLS(df_panel['birth_rate'],
                sm.add_constant(df_panel[['urbanization']]),
                entity_effects=True, time_effects=False)
res1 = mod1.fit(cov_type='clustered', cluster_entity=True)

# 模型2：加入收入
mod2 = PanelOLS(df_panel['birth_rate'],
                sm.add_constant(df_panel[['urbanization', 'log_income']]),
                entity_effects=True, time_effects=False)
res2 = mod2.fit(cov_type='clustered', cluster_entity=True)

# 模型3：双向固定效应
mod3 = PanelOLS(df_panel['birth_rate'],
                sm.add_constant(df_panel[['urbanization', 'log_income']]),
                entity_effects=True, time_effects=True)
res3 = mod3.fit(cov_type='clustered', cluster_entity=True)

print("\n模型1（省份FE，基准）：")
print(f"  城镇化率系数: {res1.params['urbanization']:.4f}  p={res1.pvalues['urbanization']:.4f}")

print("\n模型2（省份FE，加入收入）：")
print(f"  城镇化率系数: {res2.params['urbanization']:.4f}  p={res2.pvalues['urbanization']:.4f}")
print(f"  收入对数系数: {res2.params['log_income']:.4f}  p={res2.pvalues['log_income']:.4f}")

print("\n模型3（双向FE）：")
print(f"  城镇化率系数: {res3.params['urbanization']:.4f}  p={res3.pvalues['urbanization']:.4f}")
print(f"  收入对数系数: {res3.params['log_income']:.4f}  p={res3.pvalues['log_income']:.4f}")

# 回归系数可视化
fig, ax = plt.subplots(figsize=(9, 5))
models = ['模型1\n（省份FE）', '模型2\n（省份FE+收入）', '模型3\n（双向FE）']
coefs_urb = [res1.params['urbanization'], res2.params['urbanization'], res3.params['urbanization']]
cis_urb = [
    (res1.conf_int().loc['urbanization','lower'], res1.conf_int().loc['urbanization','upper']),
    (res2.conf_int().loc['urbanization','lower'], res2.conf_int().loc['urbanization','upper']),
    (res3.conf_int().loc['urbanization','lower'], res3.conf_int().loc['urbanization','upper']),
]
x_pos = range(len(models))
ax.bar(x_pos, coefs_urb, color='#3498db', alpha=0.7, width=0.5)
for i, (coef, (lo, hi)) in enumerate(zip(coefs_urb, cis_urb)):
    ax.errorbar(i, coef, yerr=[[coef-lo], [hi-coef]], fmt='none', color='black', capsize=5)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_title('城镇化率对出生率的影响系数（95%置信区间）', fontsize=12)
ax.set_ylabel('回归系数')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_panel_coefs.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ 图3已保存：fig3_panel_coefs.png")

# ============================================================
# 第四步：分位数回归（异质性分析）
# ============================================================
print("\n" + "=" * 60)
print("第四步：分位数回归（区域异质性）")
print("=" * 60)

import statsmodels.formula.api as smf

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
qr_results = []

for q in quantiles:
    qr = smf.quantreg('birth_rate ~ urbanization + log_income + year_trend', data=df).fit(q=q)
    qr_results.append({
        'quantile': q,
        'urb_coef': qr.params['urbanization'],
        'urb_pval': qr.pvalues['urbanization'],
        'inc_coef': qr.params['log_income'],
        'inc_pval': qr.pvalues['log_income'],
    })

qr_df = pd.DataFrame(qr_results)
print(qr_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(qr_df['quantile'], qr_df['urb_coef'], 'b-o', linewidth=2)
axes[0].axhline(y=res3.params['urbanization'], color='red', linestyle='--',
                label='OLS估计')
axes[0].fill_between(qr_df['quantile'],
                     qr_df['urb_coef'] * 0.85,
                     qr_df['urb_coef'] * 1.15,
                     alpha=0.2, color='blue')
axes[0].set_title('城镇化率系数的分位数变化', fontsize=12)
axes[0].set_xlabel('分位数')
axes[0].set_ylabel('回归系数')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(qr_df['quantile'], qr_df['inc_coef'], 'g-o', linewidth=2)
axes[1].axhline(y=res3.params['log_income'], color='red', linestyle='--',
                label='OLS估计')
axes[1].fill_between(qr_df['quantile'],
                     qr_df['inc_coef'] * 1.15,
                     qr_df['inc_coef'] * 0.85,
                     alpha=0.2, color='green')
axes[1].set_title('收入水平系数的分位数变化', fontsize=12)
axes[1].set_xlabel('分位数')
axes[1].set_ylabel('回归系数')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig4_quantile.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ 图4已保存：fig4_quantile.png")

# ============================================================
# 第五步：稳健性检验 - 区域分组回归
# ============================================================
print("\n" + "=" * 60)
print("第五步：稳健性检验 - 分区域回归")
print("=" * 60)

robustness = []
for region in ['东部', '中部', '西部']:
    sub = df[df['region'] == region].set_index(['province', 'year'])
    try:
        m = PanelOLS(sub['birth_rate'],
                     sm.add_constant(sub[['urbanization', 'log_income']]),
                     entity_effects=True)
        r = m.fit(cov_type='clustered', cluster_entity=True)
        robustness.append({
            '区域': region,
            '城镇化率系数': round(r.params['urbanization'], 4),
            '城镇化率p值': round(r.pvalues['urbanization'], 4),
            '收入系数': round(r.params['log_income'], 4),
            '收入p值': round(r.pvalues['log_income'], 4),
        })
        print(f"  {region}: 城镇化={r.params['urbanization']:.4f}(p={r.pvalues['urbanization']:.4f}), "
              f"收入={r.params['log_income']:.4f}(p={r.pvalues['log_income']:.4f})")
    except Exception as e:
        print(f"  {region} 分组样本较少，跳过: {e}")

rob_df = pd.DataFrame(robustness)

# ============================================================
# 输出回归结果表（用于论文）
# ============================================================
print("\n" + "=" * 60)
print("📊 主回归结果汇总表（填入论文）")
print("=" * 60)

def stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.1: return '*'
    return ''

print(f"\n{'变量':<15} {'模型1':>15} {'模型2':>15} {'模型3（双向FE）':>15}")
print("-" * 65)
print(f"{'城镇化率':<15} "
      f"{res1.params['urbanization']:.4f}{stars(res1.pvalues['urbanization']):>4}"
      f"{res2.params['urbanization']:.4f}{stars(res2.pvalues['urbanization']):>4}"
      f"{res3.params['urbanization']:.4f}{stars(res3.pvalues['urbanization']):>4}")
print(f"{'收入对数':<15} "
      f"{'—':>19}"
      f"{res2.params['log_income']:.4f}{stars(res2.pvalues['log_income']):>4}"
      f"{res3.params['log_income']:.4f}{stars(res3.pvalues['log_income']):>4}")
print("-" * 65)
print(f"{'省份固定效应':<15} {'YES':>15} {'YES':>15} {'YES':>15}")
print(f"{'时间固定效应':<15} {'NO':>15} {'NO':>15} {'YES':>15}")
print(f"{'观测值':<15} {res1.nobs:>15.0f} {res2.nobs:>15.0f} {res3.nobs:>15.0f}")
print(f"{'R²（组内）':<15} {res1.rsquared:>15.4f} {res2.rsquared:>15.4f} {res3.rsquared:>15.4f}")
print("注：***p<0.01，**p<0.05，*p<0.1，聚类标准误（省份层面）")

# 保存分位数结果
qr_df.to_csv('quantile_results.csv', index=False, encoding='utf-8-sig')
rob_df.to_csv('robustness_results.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("✅ 所有建模完成！")
print("生成文件：")
print("  fig1_trend.png      — 出生率趋势图")
print("  fig2_xgboost.png    — 变量重要性图")
print("  fig3_panel_coefs.png — 面板回归系数图")
print("  fig4_quantile.png   — 分位数回归图")
print("  quantile_results.csv — 分位数回归结果")
print("  robustness_results.csv — 稳健性检验结果")
print("=" * 60)
