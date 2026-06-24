import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. 基本参数
# ============================================================

np.random.seed(42)

width, height = 512, 512
n_points = 12000
n_times = 80

t = np.linspace(0, 1, n_times)

x = np.random.uniform(0, width, n_points)
y = np.random.uniform(0, height, n_points)

deformation = np.zeros((n_points, n_times))
labels = np.full(n_points, "Stable", dtype=object)

# 新增：非线性强度等级
# Medium   : 中等非线性
# Critical : 临界非线性
# Strong   : 强非线性
nonlinear_level = np.full(n_points, "None", dtype=object)
nonlinear_residual_target_mm = np.zeros(n_points, dtype=float)


# ============================================================
# 2. 圆形区域和高斯权重函数
# ============================================================

def circle_mask(x, y, xc, yc, radius):
    return (x - xc) ** 2 + (y - yc) ** 2 <= radius ** 2


def gaussian_weight(x, y, xc, yc, sigma):
    r2 = (x - xc) ** 2 + (y - yc) ** 2
    return np.exp(-r2 / (2 * sigma ** 2))


def best_linear_residual_scale(basis, t):
    """
    计算一个非线性基函数 basis(t) 去除最佳线性趋势 beta*t 后的最大残差。

    若 deformation = C * basis(t)，则去除最佳线性趋势后的最大非线性残差为：
        C * max(|basis(t) - beta*t|)

    这里返回 max(|basis(t) - beta*t|)，用于把不同类型形变统一控制到
    指定的非线性残差等级。
    """
    basis = np.asarray(basis, dtype=float)

    beta = np.dot(t, basis) / np.dot(t, t)
    residual = basis - beta * t

    scale = np.max(np.abs(residual))

    if scale < 1e-12:
        scale = 1.0

    return scale


# def assign_level_by_weight(mask, weight):
#     """
#     在某一种非线性形变区域内部，按照高斯权重从小到大排序，
#     分成三等份：

#         外圈/较弱权重  -> Medium
#         中间权重      -> Critical
#         中心/较强权重 -> Strong

#     这样可以保证每一种非线性类型内部：
#         中等 : 临界 : 强非线性 ≈ 1 : 1 : 1
#     """
#     idx = np.where(mask)[0]

#     order = idx[np.argsort(weight[idx])]

#     medium_idx, critical_idx, strong_idx = np.array_split(order, 3)

#     nonlinear_level[medium_idx] = "Medium"
#     nonlinear_level[critical_idx] = "Critical"
#     nonlinear_level[strong_idx] = "Strong"

#     return medium_idx, critical_idx, strong_idx

def assign_level_by_weight(mask, weight):
    """
    在某一种非线性形变区域内部，按照高斯权重从小到大排序，
    分成三档：

        外圈/较弱权重  -> Medium
        中间权重      -> Critical
        中心/较强权重 -> Strong

    比例设置为：

        Medium : Critical : Strong = 1 : 1 : 2

    也就是：

        Medium + Critical : Strong = 1 : 1

    这样可以保证：
        50% 点为中等或临界非线性；
        50% 点为强非线性。
    """
    idx = np.where(mask)[0]

    order = idx[np.argsort(weight[idx])]

    n = len(order)

    n_medium = n // 4
    n_critical = n // 4

    medium_idx = order[:n_medium]
    critical_idx = order[n_medium:n_medium + n_critical]
    strong_idx = order[n_medium + n_critical:]

    nonlinear_level[medium_idx] = "Medium"
    nonlinear_level[critical_idx] = "Critical"
    nonlinear_level[strong_idx] = "Strong"

    return medium_idx, critical_idx, strong_idx


def assign_target_residual(level_groups):
    """
    为三档非线性分配目标残差，单位 mm。

    这里的残差不是首末累计形变量，而是：
        真实形变序列 - 最佳线性拟合序列
    之后的最大残差。

    Sentinel-1 C 波段四分之一波长约为 14 mm，
    所以这里设置：

        Medium   : 5  - 10 mm
        Critical : 10 - 15 mm
        Strong   : 15 - 30 mm
    """
    medium_idx, critical_idx, strong_idx = level_groups

    target = np.zeros(n_points, dtype=float)

    target[medium_idx] = np.random.uniform(
        5.0,
        10.0,
        size=len(medium_idx)
    )

    target[critical_idx] = np.random.uniform(
        10.0,
        15.0,
        size=len(critical_idx)
    )

    target[strong_idx] = np.random.uniform(
        15.0,
        35.0,
        size=len(strong_idx)
    )

    nonlinear_residual_target_mm[medium_idx] = target[medium_idx]
    nonlinear_residual_target_mm[critical_idx] = target[critical_idx]
    nonlinear_residual_target_mm[strong_idx] = target[strong_idx]

    return target


# ============================================================
# 3. 设置五个圆形形变区域
# ============================================================

center_quad = (128, 384)      # 左上：二次非线性
center_season = (384, 384)    # 右上：季节性
center_jump = (128, 128)      # 左下：突变型
center_piece = (384, 128)     # 右下：分段线性
center_power = (256, 256)     # 中间：幂律形变

radius = 85
radius_power = 78

mask_quad = circle_mask(x, y, *center_quad, radius)
mask_season = circle_mask(x, y, *center_season, radius)
mask_jump = circle_mask(x, y, *center_jump, radius)
mask_piece = circle_mask(x, y, *center_piece, radius)
mask_power = circle_mask(x, y, *center_power, radius_power)

mask_nonlinear = (
    mask_quad |
    mask_season |
    mask_jump |
    mask_piece |
    mask_power
)

mask_background = ~mask_nonlinear

linear_probability = 0.45
random_background = np.random.rand(n_points)

mask_linear = mask_background & (random_background < linear_probability)
mask_stable = mask_background & (random_background >= linear_probability)


# ============================================================
# 4. 生成不同类型形变时间序列
# ============================================================

# ------------------------------------------------------------
# 左上：二次非线性
# d(t) = -C * t^2
# 控制的是去除最佳线性趋势后的最大非线性残差
# ------------------------------------------------------------

w_quad = gaussian_weight(
    x,
    y,
    center_quad[0],
    center_quad[1],
    sigma=radius / 2.2
)

groups_quad = assign_level_by_weight(mask_quad, w_quad)
target_quad = assign_target_residual(groups_quad)

basis_quad = t ** 2
scale_quad = best_linear_residual_scale(basis_quad, t)

for i in np.where(mask_quad)[0]:
    C = target_quad[i] / scale_quad
    deformation[i, :] = -C * basis_quad

labels[mask_quad] = "Quadratic"


# ------------------------------------------------------------
# 右上：季节性形变
# d(t) = trend*t + C*sin(2*pi*t + phase)
# 控制的是正弦项去除最佳线性趋势后的最大非线性残差
# ------------------------------------------------------------

w_season = gaussian_weight(
    x,
    y,
    center_season[0],
    center_season[1],
    sigma=radius / 2.2
)

groups_season = assign_level_by_weight(mask_season, w_season)
target_season = assign_target_residual(groups_season)

trend_season = -8 * gaussian_weight(
    x,
    y,
    center_season[0],
    center_season[1],
    sigma=radius / 1.8
)

phase = 2 * np.pi * (x / width + y / height)

for i in np.where(mask_season)[0]:
    basis_season = np.sin(2 * np.pi * t + phase[i])
    scale_season = best_linear_residual_scale(basis_season, t)

    C = target_season[i] / scale_season

    deformation[i, :] = trend_season[i] * t + C * basis_season

labels[mask_season] = "Seasonal"


# ------------------------------------------------------------
# 左下：突变型形变
#
# 修改后模型：
#
#   t < t0:
#       d(t) = v1 * t
#
#   t >= t0:
#       d(t) = v1 * t0 - S + v2 * (t - t0)
#
# 其中：
#   v1：突变前线性沉降速度
#   S ：突变幅度
#   v2：突变后线性沉降速度
#
# 这样生成的突变型形变在突变前后都有线性沉降，
# 中间叠加一次瞬时阶跃。
# ------------------------------------------------------------

w_jump = gaussian_weight(
    x,
    y,
    center_jump[0],
    center_jump[1],
    sigma=radius / 2.2
)

groups_jump = assign_level_by_weight(mask_jump, w_jump)

medium_idx, critical_idx, strong_idx = groups_jump

jump_amplitude = np.zeros(n_points, dtype=float)

# 中等阶跃：传统方法大概率可以正确解缠
jump_amplitude[medium_idx] = np.random.uniform(
    5.0,
    10.0,
    size=len(medium_idx)
)

# 临界阶跃：传统方法部分可以正确解缠，部分容易出现整数模糊度错误
jump_amplitude[critical_idx] = np.random.uniform(
    10.0,
    15.0,
    size=len(critical_idx)
)

# 强阶跃：传统方法更容易出错，但幅度仍控制在所提方法可处理范围内
jump_amplitude[strong_idx] = np.random.uniform(
    15.0,
    22.0,
    size=len(strong_idx)
)

# 记录阶跃类型的非线性强度，方便后面统计和画图
nonlinear_residual_target_mm[medium_idx] = jump_amplitude[medium_idx]
nonlinear_residual_target_mm[critical_idx] = jump_amplitude[critical_idx]
nonlinear_residual_target_mm[strong_idx] = jump_amplitude[strong_idx]

# 阶跃发生时间
t0 = 0.52

# 突变前后的线性沉降速度
# 注意：因为 t 是 0 到 1，所以这里的速度可以理解为
# 一个完整观测周期内的累计线性形变量，单位 mm。
v1_jump = np.zeros(n_points, dtype=float)
v2_jump = np.zeros(n_points, dtype=float)

# 突变前线性沉降
v1_jump[medium_idx] = np.random.uniform(
    -6.0,
    -3.0,
    size=len(medium_idx)
)

v1_jump[critical_idx] = np.random.uniform(
    -8.0,
    -4.0,
    size=len(critical_idx)
)

v1_jump[strong_idx] = np.random.uniform(
    -10.0,
    -5.0,
    size=len(strong_idx)
)

# 突变后线性沉降
# 这里设置得略大一些，表示突变后仍持续沉降
v2_jump[medium_idx] = np.random.uniform(
    -8.0,
    -4.0,
    size=len(medium_idx)
)

v2_jump[critical_idx] = np.random.uniform(
    -10.0,
    -5.0,
    size=len(critical_idx)
)

v2_jump[strong_idx] = np.random.uniform(
    -12.0,
    -6.0,
    size=len(strong_idx)
)

for i in np.where(mask_jump)[0]:

    S = jump_amplitude[i]
    v1 = v1_jump[i]
    v2 = v2_jump[i]

    deformation[i, :] = np.where(
        t < t0,
        v1 * t,
        v1 * t0 - S + v2 * (t - t0)
    )

labels[mask_jump] = "Abrupt"



# ------------------------------------------------------------
# 右下：分段线性形变
# d(t) = v1*t - C*(t-tb)*H(t-tb)
# 控制的是速度变化后逐渐累积出来的非线性残差
# ------------------------------------------------------------

w_piece = gaussian_weight(
    x,
    y,
    center_piece[0],
    center_piece[1],
    sigma=radius / 2.2
)

groups_piece = assign_level_by_weight(mask_piece, w_piece)
target_piece = assign_target_residual(groups_piece)

tb = 0.50
basis_piece = np.where(t < tb, 0.0, t - tb)
scale_piece = best_linear_residual_scale(basis_piece, t)

# 第一段速度可以保留一个较弱的线性背景
v1_piece = -6 * w_piece

for i in np.where(mask_piece)[0]:
    C = target_piece[i] / scale_piece
    deformation[i, :] = v1_piece[i] * t - C * basis_piece

labels[mask_piece] = "Piecewise"


# ------------------------------------------------------------
# 中间：幂律形变
# d(t) = -C*t^alpha
# 控制的是幂律项去除最佳线性趋势后的最大非线性残差
# ------------------------------------------------------------

w_power = gaussian_weight(
    x,
    y,
    center_power[0],
    center_power[1],
    sigma=radius_power / 2.2
)

groups_power = assign_level_by_weight(mask_power, w_power)
target_power = assign_target_residual(groups_power)

alpha_power = 0.55
basis_power = t ** alpha_power
scale_power = best_linear_residual_scale(basis_power, t)

for i in np.where(mask_power)[0]:
    C = target_power[i] / scale_power
    deformation[i, :] = -C * basis_power

labels[mask_power] = "Power-law"


# ------------------------------------------------------------
# 背景：线性形变
# ------------------------------------------------------------

v_linear = -2 - 5 * (x / width) + 3 * (y / height)

for i in np.where(mask_linear)[0]:
    deformation[i, :] = v_linear[i] * t

labels[mask_linear] = "Linear"


# ------------------------------------------------------------
# 背景：稳定型
# ------------------------------------------------------------

for i in np.where(mask_stable)[0]:
    deformation[i, :] = np.random.normal(0, 0.4, size=n_times)

labels[mask_stable] = "Stable"


# ============================================================
# 5. 加入观测噪声
# ============================================================

noise_std = 0.6

deformation_noisy = deformation + np.random.normal(
    0,
    noise_std,
    size=deformation.shape
)


# ============================================================
# 6. 统计不同类型中三档非线性数量
# ============================================================

print("\n================ 非线性等级统计 ================")

for name in ["Quadratic", "Seasonal", "Abrupt", "Piecewise", "Power-law"]:

    idx = labels == name

    print(f"\n{name}:")

    unique, counts = np.unique(nonlinear_level[idx], return_counts=True)

    for u, c in zip(unique, counts):
        print(f"  {u:8s}: {c}")


# ============================================================
# 7. 绘制第 20、40、60 个时间点的累计形变量图
# ============================================================

time_indices = [24, 49, 79]
time_labels = [25, 50, 80]

# time_indices = [19, 39, 59]
# time_labels = [20, 40, 60]

for idx, label in zip(time_indices, time_labels):

    deformation_at_time = deformation_noisy[:, idx]

    plt.figure(figsize=(7, 7))

    scatter = plt.scatter(
        x,
        y,
        c=deformation_at_time,
        s=6,
        alpha=0.88,
        cmap="jet",
        vmin=-120,
        vmax=120
    )

    cbar = plt.colorbar(scatter, shrink=0.82)
    cbar.set_label("Cumulative deformation / mm")

    plt.text(58, 455, "Quadratic", fontsize=12, weight="bold")
    plt.text(318, 455, "Seasonal", fontsize=12, weight="bold")
    plt.text(65, 55, "Abrupt", fontsize=12, weight="bold")
    plt.text(310, 55, "Piecewise", fontsize=12, weight="bold")
    plt.text(210, 265, "Power-law", fontsize=12, weight="bold")

    plt.xlabel("Range direction / pixel")
    plt.ylabel("Azimuth direction / pixel")
    plt.title(f"Simulated Cumulative Deformation Field at Time {label}")

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()

    save_name = f"deformation_time_{label:02d}.png"
    plt.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Figure saved: {save_name}")


# ============================================================
# 8. 保存 CSV 文件
# ============================================================

data = {
    "point_id": np.arange(n_points),
    "x": x,
    "y": y,
    "type": labels,
    "nonlinear_level": nonlinear_level,
    "nonlinear_residual_target_mm": nonlinear_residual_target_mm,
    "final_deformation_mm": deformation_noisy[:, -1]
}

for k in range(n_times):
    data[f"deformation_t{k+1:02d}_mm"] = deformation_noisy[:, k]

df = pd.DataFrame(data)

csv_name = "simulated_deformation_points.csv"
df.to_csv(csv_name, index=False, encoding="utf-8-sig")

print(f"CSV file saved: {csv_name}")
print(df.head())