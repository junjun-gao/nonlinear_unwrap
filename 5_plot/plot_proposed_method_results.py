import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. 输入输出路径
# ============================================================

CSV_PATH = "/data/test/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/4_result_csv/simulated_unwrapped_noise_0p2rad_result.csv"

SAVE_DEFORMATION_EST_PNG = "deformation_est.png"
SAVE_HEIGHT_EST_PNG = "height_est.png"
SAVE_DEFORMATION_MAE_PNG = "deformation_mae.png"
SAVE_HEIGHT_ABS_ERROR_PNG = "height_abs_error.png"

SAVE_DEFORMATION_EST_SVG = "deformation_est.svg"
SAVE_HEIGHT_EST_SVG = "height_est.svg"
SAVE_DEFORMATION_MAE_SVG = "deformation_mae.svg"
SAVE_HEIGHT_ABS_ERROR_SVG = "height_abs_error.svg"


# ============================================================
# 2. 参考点序号
# ============================================================

REF_POINT_ID = 11341


# ============================================================
# 3. 读取 CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

x = df["x"].values
y = df["y"].values


# ============================================================
# 4. 获取参考点坐标
# ============================================================

ref_row = df[df["point_id"] == REF_POINT_ID]

if ref_row.empty:
    raise ValueError("未找到参考点 point_id = {}".format(REF_POINT_ID))

ref_x = ref_row["x"].values[0]
ref_y = ref_row["y"].values[0]

print("Reference point ID:", REF_POINT_ID)
print("Reference point x:", ref_x)
print("Reference point y:", ref_y)


# ============================================================
# 5. 读取需要绘制的数据
# ============================================================

# 最终时刻形变量估计值，单位 mm
deformation_est = df["deformation_est_t60_mm"].values

# 高程误差估计值，单位 m
height_est = df["h_est_m_relative"].values

# 形变量 MAE
deformation_error_cols = [
    col for col in df.columns
    if col.startswith("deformation_error_t") and col.endswith("_mm")
]

deformation_mae = np.mean(
    np.abs(df[deformation_error_cols].values),
    axis=1
)

# 高程误差绝对值
height_abs_error = np.abs(df["delta_h_error_m"].values)


# ============================================================
# 6. 计算整体指标
# ============================================================

global_deformation_mae = np.mean(deformation_mae)
global_height_mae = np.mean(height_abs_error)

print("Global deformation MAE: {:.4f} mm".format(global_deformation_mae))
print("Global height error MAE: {:.4f} m".format(global_height_mae))


# ============================================================
# 7. 绘图函数：单独保存一张图
# ============================================================

def save_single_scatter_map(
    x,
    y,
    value,
    cbar_label,
    save_png,
    save_svg,
    ref_x,
    ref_y,
    vmin=None,
    vmax=None,
    cmap="jet"
):
    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(
        x,
        y,
        c=value,
        s=6,
        alpha=0.88,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ========================================================
    # 标记参考点：黑色三角形
    # ========================================================
    ax.scatter(
        ref_x,
        ref_y,
        s=45,
        marker="^",
        c="black",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
        label="reference point"
    )

    # 图例放在右下角
    ax.legend(
        loc="lower right",
        fontsize=9,
        frameon=True
    )

    # 不设置图标题
    ax.set_xlabel("Range direction / pixel", fontsize=10)
    ax.set_ylabel("Azimuth direction / pixel", fontsize=10)

    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=9)

    plt.tight_layout()

    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_svg, bbox_inches="tight")

    plt.close()

    print("Figure saved:", save_png)
    print("Figure saved:", save_svg)


# ============================================================
# 8. 设置色带范围
# ============================================================

defo_vmin = -50
defo_vmax = 50

height_vmin = np.nanpercentile(height_est, 2)
height_vmax = np.nanpercentile(height_est, 98)

defo_mae_vmin = 0
defo_mae_vmax = np.nanpercentile(deformation_mae, 98)

height_mae_vmin = 0
height_mae_vmax = np.nanpercentile(height_abs_error, 98)


# ============================================================
# 9. 分别输出 4 张图
# ============================================================

save_single_scatter_map(
    x=x,
    y=y,
    value=deformation_est,
    cbar_label="Cumulative deformation / mm",
    save_png=SAVE_DEFORMATION_EST_PNG,
    save_svg=SAVE_DEFORMATION_EST_SVG,
    ref_x=ref_x,
    ref_y=ref_y,
    vmin=defo_vmin,
    vmax=defo_vmax
)

save_single_scatter_map(
    x=x,
    y=y,
    value=height_est,
    cbar_label="Residual topographic error / m",
    save_png=SAVE_HEIGHT_EST_PNG,
    save_svg=SAVE_HEIGHT_EST_SVG,
    ref_x=ref_x,
    ref_y=ref_y,
    vmin=height_vmin,
    vmax=height_vmax
)

save_single_scatter_map(
    x=x,
    y=y,
    value=deformation_mae,
    cbar_label="MAE / mm",
    save_png=SAVE_DEFORMATION_MAE_PNG,
    save_svg=SAVE_DEFORMATION_MAE_SVG,
    ref_x=ref_x,
    ref_y=ref_y,
    vmin=defo_mae_vmin,
    vmax=defo_mae_vmax
)

save_single_scatter_map(
    x=x,
    y=y,
    value=height_abs_error,
    cbar_label="Absolute error / m",
    save_png=SAVE_HEIGHT_ABS_ERROR_PNG,
    save_svg=SAVE_HEIGHT_ABS_ERROR_SVG,
    ref_x=ref_x,
    ref_y=ref_y,
    vmin=height_mae_vmin,
    vmax=height_mae_vmax
)