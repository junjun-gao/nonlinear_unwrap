# -*- coding: utf-8 -*-
"""
绘制真实形变量 vs 仿真估计形变量 的密度散点图

横坐标：真实形变量 true deformation，单位 mm
纵坐标：估计形变量 estimated deformation，单位 mm
颜色：该位置附近点的数量

适用于包含如下字段的 CSV：
    deformation_true_relative_t01_mm ... deformation_true_relative_t60_mm
    deformation_est_t01_mm ... deformation_est_t60_mm

如果没有 deformation_true_relative_tXX_mm，
则自动使用 deformation_tXX_mm 作为真实值。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ============================================================
# 1. 输入输出路径
# ============================================================

CSV_PATH = "/data/test/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/4_result_csv/simulated_unwrapped_grid_noise_0p2rad_result.csv"   # 修改为你的 CSV 文件名

SAVE_PNG = "true_vs_est_density_scatter.png"
SAVE_SVG = "true_vs_est_density_scatter.svg"


# ============================================================
# 2. 基本参数
# ============================================================

N_TIMES = 60

# 二维统计网格数量，越大越细，越小越平滑
BINS = 250

# 散点大小
POINT_SIZE = 2.0

# 散点透明度
POINT_ALPHA = 0.75

# 是否使用对数颜色，更适合点数很多的情况
USE_LOG_COLOR = True


# ============================================================
# 3. 读取 CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("读取 CSV 完成:", CSV_PATH)
print("点数:", len(df))


# ============================================================
# 4. 自动确定真实值字段和估计值字段
# ============================================================

true_relative_cols = [
    "deformation_true_relative_t{:02d}_mm".format(i)
    for i in range(1, N_TIMES + 1)
]

true_original_cols = [
    "deformation_t{:02d}_mm".format(i)
    for i in range(1, N_TIMES + 1)
]

est_cols = [
    "deformation_est_t{:02d}_mm".format(i)
    for i in range(1, N_TIMES + 1)
]


# 优先使用相对真实形变量，因为估计结果通常是相对参考点的结果
if all(col in df.columns for col in true_relative_cols):
    true_cols = true_relative_cols
    true_name = "relative true deformation"
    print("真实值字段: deformation_true_relative_tXX_mm")

elif all(col in df.columns for col in true_original_cols):
    true_cols = true_original_cols
    true_name = "true deformation"
    print("真实值字段: deformation_tXX_mm")

else:
    raise ValueError("CSV 中没有找到完整的真实形变量字段。")


if not all(col in df.columns for col in est_cols):
    raise ValueError("CSV 中没有找到完整的估计形变量字段 deformation_est_tXX_mm。")


# ============================================================
# 5. 展开所有点、所有时相
# ============================================================

true_arr = df[true_cols].values.astype(float)
est_arr = df[est_cols].values.astype(float)

# shape: n_points × n_times 变为一维
x_true = true_arr.reshape(-1)
y_est = est_arr.reshape(-1)

# 去除 NaN 或 inf
valid = np.isfinite(x_true) & np.isfinite(y_est)
x_true = x_true[valid]
y_est = y_est[valid]

print("有效散点数量:", len(x_true))


# ============================================================
# 6. 计算每个散点所在位置的密度 / 数量
# ============================================================

# 计算二维直方图
counts, x_edges, y_edges = np.histogram2d(
    x_true,
    y_est,
    bins=BINS
)

# 找到每个点所在的网格编号
x_bin = np.searchsorted(x_edges, x_true, side="right") - 1
y_bin = np.searchsorted(y_edges, y_est, side="right") - 1

# 防止边界越界
x_bin = np.clip(x_bin, 0, BINS - 1)
y_bin = np.clip(y_bin, 0, BINS - 1)

# 每个点对应的密度，也就是所在网格中的点数
density = counts[x_bin, y_bin]

# 为了让高密度点绘制在上层，按 density 排序
order = np.argsort(density)
x_true = x_true[order]
y_est = y_est[order]
density = density[order]


# ============================================================
# 7. 计算评价指标
# ============================================================

error = y_est - x_true

mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error ** 2))
bias = np.mean(error)

ss_res = np.sum((y_est - x_true) ** 2)
ss_tot = np.sum((x_true - np.mean(x_true)) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

print("\n================ 精度评价 ================")
print("MAE  : {:.4f} mm".format(mae))
print("RMSE : {:.4f} mm".format(rmse))
print("Bias : {:.4f} mm".format(bias))
print("R^2  : {:.4f}".format(r2))


# ============================================================
# 8. 绘图
# ============================================================

fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=300)

if USE_LOG_COLOR:
    norm = LogNorm(vmin=max(1, np.min(density[density > 0])), vmax=np.max(density))
else:
    norm = None

sc = ax.scatter(
    x_true,
    y_est,
    c=density,
    s=POINT_SIZE,
    alpha=POINT_ALPHA,
    cmap="jet",
    norm=norm,
    edgecolors="none"
)

# 绘制 y = x 理想线
min_val = min(np.min(x_true), np.min(y_est))
max_val = max(np.max(x_true), np.max(y_est))

padding = 0.05 * (max_val - min_val)
plot_min = min_val - padding
plot_max = max_val + padding

ax.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    "k--",
    linewidth=1.2,
    label="y = x"
)

ax.set_xlim(plot_min, plot_max)
ax.set_ylim(plot_min, plot_max)

ax.set_xlabel("True deformation / mm", fontsize=12)
ax.set_ylabel("Estimated deformation / mm", fontsize=12)

ax.set_title(
    "True vs Estimated Deformation",
    fontsize=13,
)

# 坐标轴比例设为一致，避免 y=x 线视觉上不是 45°
ax.set_aspect("equal", adjustable="box")

# 颜色条
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Number of points", fontsize=11)

# 左上角写指标
text = (
    "MAE = {:.3f} mm\n"
    "RMSE = {:.3f} mm\n"
    "Bias = {:.3f} mm\n"
    "$R^2$ = {:.3f}"
).format(mae, rmse, bias, r2)

ax.text(
    0.04,
    0.96,
    text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.75,
        edgecolor="none"
    )
)

ax.legend(loc="lower right", fontsize=10, frameon=True)

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

plt.tight_layout()

plt.savefig(SAVE_PNG, dpi=300, bbox_inches="tight")
plt.savefig(SAVE_SVG, bbox_inches="tight")

plt.close()

print("\n图片已保存:")
print(SAVE_PNG)
print(SAVE_SVG)