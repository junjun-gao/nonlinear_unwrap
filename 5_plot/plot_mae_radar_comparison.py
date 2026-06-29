import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


# ============================================================
# 1. 输入数据
# ============================================================

deformation_types = [
    "Quadratic",
    "piecewise\nlinear",
    "Seasonal",
    "power-\nlaw",
    "Step"
]

# MAE_d 数据，单位 mm
# mae_proposed = np.array([1.618125, 2.741008, 1.771017, 1.827552, 1.517604])
# mae_traditional = np.array([2.642588, 3.934056, 5.618529, 8.849339, 2.103522])

# # # # MAE_h 数据，单位 mm
# mae_proposed = np.array([2.434746, 2.409406, 3.753295, 2.550401, 2.452542])
# mae_traditional = np.array([4.996617, 6.173325, 5.821925, 5.286213, 4.364520])

# # RMSE_d 数据，单位 mm
# mae_proposed = np.array([2.338330, 3.495418, 2.423864, 2.719965, 2.092527])
# mae_traditional = np.array([3.655635, 5.496923, 8.111480, 12.476094, 2.631484])

# RMSE_h 数据，单位 mm
mae_proposed = np.array([3.386360, 3.349236, 4.571465, 3.285897, 3.158737])
mae_traditional = np.array([6.370578, 7.679457, 7.289123, 6.754058, 5.561379])


# ============================================================
# 2. 设置 MAE 显示范围
# ============================================================

mae_min = 0.0
mae_max = 10.0

mae_proposed_plot = np.clip(mae_proposed, mae_min, mae_max)
mae_traditional_plot = np.clip(mae_traditional, mae_min, mae_max)


# ============================================================
# 3. 设置五边形角度
# ============================================================

n = len(deformation_types)

# 从正上方开始，顺时针排列
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
angles = np.pi / 2 - angles

unit_x = np.cos(angles)
unit_y = np.sin(angles)


# ============================================================
# 4. MAE 映射到半径
# ============================================================

def mae_to_radius(mae):
    """
    0 mm  -> 半径 0
    10 mm -> 半径 1
    """
    return mae / mae_max


def get_polygon_xy(values):
    r = mae_to_radius(values)

    x = r * unit_x
    y = r * unit_y

    x = np.concatenate([x, [x[0]]])
    y = np.concatenate([y, [y[0]]])

    return x, y


# ============================================================
# 5. 开始绘图
# ============================================================

fig, ax = plt.subplots(figsize=(6.2, 6.2))

ax.set_aspect("equal")
ax.axis("off")


# ============================================================
# 6. 五边形背景渐变
# ============================================================

outer_x = unit_x
outer_y = unit_y

outer_x_closed = np.concatenate([outer_x, [outer_x[0]]])
outer_y_closed = np.concatenate([outer_y, [outer_y[0]]])

# 构造五边形裁剪路径
vertices = np.column_stack([outer_x_closed, outer_y_closed])
codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
pentagon_path = Path(vertices, codes)
pentagon_patch = PathPatch(pentagon_path, facecolor="none", edgecolor="none")
ax.add_patch(pentagon_patch)

# 渐变背景：左下偏淡黄，右上偏淡绿
grid_size = 500
xx, yy = np.meshgrid(
    np.linspace(-1.05, 1.05, grid_size),
    np.linspace(-1.05, 1.05, grid_size)
)

gradient = (xx + yy)
gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())

# RGB 渐变颜色
# 淡黄色 -> 淡绿色
color_yellow = np.array([1.00, 0.96, 0.60])
color_green = np.array([0.78, 0.95, 0.72])

rgb = color_yellow * (1 - gradient[..., None]) + color_green * gradient[..., None]

im = ax.imshow(
    rgb,
    extent=[-1.05, 1.05, -1.05, 1.05],
    origin="lower",
    zorder=0
)
im.set_clip_path(pentagon_patch)


# ============================================================
# 7. 绘制五边形网格线，单位 mm
# ============================================================

grid_values = [2, 4, 6, 8, 10]

for gv in grid_values:
    r = gv / mae_max

    gx = r * unit_x
    gy = r * unit_y

    gx = np.concatenate([gx, [gx[0]]])
    gy = np.concatenate([gy, [gy[0]]])

    ax.plot(
        gx,
        gy,
        color="gray",
        linewidth=0.8,
        alpha=0.75,
        zorder=1
    )


# ============================================================
# 8. 绘制从中心到五个顶点的径向线
# ============================================================

for i in range(n):
    ax.plot(
        [0, unit_x[i]],
        [0, unit_y[i]],
        color="gray",
        linewidth=0.8,
        alpha=0.75,
        zorder=1
    )


# ============================================================
# 9. 添加 mm 刻度
# ============================================================

for gv in grid_values:
    r = gv / mae_max

    ax.text(
        0.06,
        r,
        "{} mm".format(gv),
        fontsize=10,
        ha="left",
        va="center",
        zorder=5
    )


# ============================================================
# 10. 添加五个顶点标签
# ============================================================

label_radius = 1.18

for i, label in enumerate(deformation_types):
    lx = label_radius * unit_x[i]
    ly = label_radius * unit_y[i]

    if unit_x[i] > 0.2:
        ha = "left"
    elif unit_x[i] < -0.2:
        ha = "right"
    else:
        ha = "center"

    if unit_y[i] > 0.2:
        va = "bottom"
    elif unit_y[i] < -0.2:
        va = "top"
    else:
        va = "center"

    ax.text(
        lx,
        ly,
        label,
        fontsize=12,
        ha=ha,
        va=va,
        zorder=5
    )


# ============================================================
# 11. 绘制两种方法的 MAE 结果
# ============================================================

x_proposed, y_proposed = get_polygon_xy(mae_proposed_plot)
x_traditional, y_traditional = get_polygon_xy(mae_traditional_plot)

# 传统方法：绿色三角形
ax.plot(
    x_traditional,
    y_traditional,
    color="#2ca25f",
    linewidth=2.0,
    marker="^",
    markersize=7,
    markerfacecolor="#2ca25f",
    markeredgecolor="#2ca25f",
    label="Traditional method",
    zorder=4
)

# 所提方法：蓝色圆形
ax.plot(
    x_proposed,
    y_proposed,
    color="#1f78b4",
    linewidth=2.0,
    marker="o",
    markersize=7,
    markerfacecolor="#1f78b4",
    markeredgecolor="#1f78b4",
    label="Proposed method",
    zorder=5
)

# 可以只给所提方法加一点填充
ax.fill(
    x_proposed,
    y_proposed,
    color="#1f78b4",
    alpha=0.10,
    zorder=2
)


# ============================================================
# 12. 图例
# ============================================================

ax.legend(
    loc="upper left",
    bbox_to_anchor=(-0.06, 1.08),
    frameon=False,
    fontsize=11,
    handlelength=1.8,
    handletextpad=0.4
)


# ============================================================
# 13. 控制显示范围
# ============================================================

ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.25, 1.35)


# ============================================================
# 14. 保存
# ============================================================

plt.savefig("mae_pentagon_radar_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("mae_pentagon_radar_comparison.svg", bbox_inches="tight")

plt.close()

print("Saved:")
print("  mae_pentagon_radar_comparison.png")
print("  mae_pentagon_radar_comparison.svg")