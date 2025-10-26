# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==================== 数据准备 Data Preparation ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据 Raw data
raw_data = {
    "NH₄Cl 浓度 (mg/L)": [0, 0, 0, 3, 3, 3, 5, 5, 5, 8, 8, 8],
    "孔号": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
    "存活数": [20, 20, 20, 19, 20, 20, 17, 19, 18, 15, 11, 9],
    "子代数": [1, 2, 2, 1, 0, 0, 1, 1, 0, 0, 2, 0],
    "死亡数": [0, 0, 0, 1, 0, 0, 3, 1, 2, 5, 9, 11],
    "携卵": [3, 4, 2, 2, 1, 3, 0, 1, 1, 0, 0, 0]
}
df_raw = pd.DataFrame(raw_data)

# ==================== 指标计算 Metric Calculation ====================
def calculate_metrics(group):
    """计算各浓度组的生物学指标"""
    return pd.Series({
        "存活率": group["存活数"].mean() / 20 * 100,
        "死亡率": group["死亡数"].mean() / 20 * 100,
        "繁殖率": group["子代数"].sum() / group["存活数"].sum() * 100,
        "产卵率": group["携卵"].sum() / group["存活数"].sum() * 100
    })

# 关键修复：使用 groupby + agg 分步计算均值和标准差
df_mean = df_raw.groupby("NH₄Cl 浓度 (mg/L)").apply(calculate_metrics)
df_std = df_raw.groupby("NH₄Cl 浓度 (mg/L)").apply(calculate_metrics).std()

# 合并数据（确保列名一一对应）
df = pd.DataFrame({
    "浓度(mg/L)": df_mean.index,
    "存活率": df_mean["存活率"],
    "存活率_std": df_std["存活率"],
    "死亡率": df_mean["死亡率"],
    "死亡率_std": df_std["死亡率"],
    "繁殖率": df_mean["繁殖率"],
    "繁殖率_std": df_std["繁殖率"],
    "产卵率": df_mean["产卵率"],
    "产卵率_std": df_std["产卵率"]
})

# ==================== 绘图 Visualization ====================
sns.set(style="whitegrid", font="SimHei")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ---- 图1: 存活率与死亡率 Survival vs Mortality ----
axes[0,0].errorbar(
    df["浓度(mg/L)"], df["存活率"],
    yerr=df["存活率_std"], fmt='bo-', label="存活率/Survival"
)
axes[0,0].errorbar(
    df["浓度(mg/L)"], df["死亡率"],
    yerr=df["死亡率_std"], fmt='r^--', label="死亡率/Mortality"
)
axes[0,0].set_title("存活率与死亡率趋势\nSurvival & Mortality Trends", fontweight='bold')
axes[0,0].set_xlabel(r"$NH_4$Cl浓度(mg/L)\nConcentration", fontweight='bold')
axes[0,0].set_ylabel("百分比(%)\nPercentage", fontweight='bold')
axes[0,0].legend()

# ---- 图2: 繁殖与产卵抑制 Reproduction & Egg-bearing ----
axes[0,1].bar(
    df["浓度(mg/L)"], df["繁殖率"],
    width=1.5, alpha=0.7, label="繁殖率/Reproduction"
)
axes[0,1].bar(
    df["浓度(mg/L)"], df["产卵率"],
    width=1.5, bottom=df["繁殖率"], alpha=0.7, label="产卵率/Egg-Bearing"
)
axes[0,1].set_title("繁殖与产卵抑制效应\nReproductive Inhibition", fontweight='bold')
axes[0,1].set_ylabel("百分比(%)\nPercentage", fontweight='bold')
axes[0,1].legend()

# ---- 图3: 剂量-效应曲线（线性拟合）Dose-Response (Linear) ----
sns.regplot(
    x="浓度(mg/L)", y="存活率", data=df,
    ax=axes[1,0], order=1,  # 线性拟合避免警告
    ci=95, scatter_kws={"s": 100, "color": "#2b8cbe"},
    line_kws={"color": "#d62728", "lw":2}
)
axes[1,0].set_title("存活率剂量-效应关系\nDose-Response Relationship", fontweight='bold')
axes[1,0].set_ylabel("存活率(%)\nSurvival Rate", fontweight='bold')

# ---- 图4: 产卵率热图 Egg-Bearing Heatmap ----
heatmap_data = df_raw.pivot_table(
    index="NH₄Cl 浓度 (mg/L)", columns="孔号", values="携卵", aggfunc="mean"
)
sns.heatmap(
    heatmap_data, annot=True, cmap="Blues",
    ax=axes[1,1], cbar_kws={"label": "产卵数/Egg Count"}
)
axes[1,1].set_title("产卵率浓度依赖\nConcentration Dependency", fontweight='bold')

plt.tight_layout()

plt.show()