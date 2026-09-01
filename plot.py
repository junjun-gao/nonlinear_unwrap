import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 参数
# ============================================================
CSV_FILE = "/data/tests/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/bspline_sim_seasonal/seasonal_periodogram_results/unwrapped_deformation_noise_0p2rad.csv"
OUTPUT_DIR = Path("plot_epoch80_results")
EPOCH = 80

POINT_SIZE = 5
REF_SIZE = 45
SELECTED_POINT_SIZE = 80

# 三个目标点的大致位置
TARGET_POINTS = [
    ("P1", 116.0514, 39.9602),
    ("P2", 116.2057, 39.9271),
    ("P3", 116.0127, 39.8056)
]

# ============================================================
# 读取数据
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILE)

longitude = df["longitude"].values.astype(float)
latitude = df["latitude"].values.astype(float)

# 第80相相对于参考点的真实形变和估计形变
deformation_true = df["deformation_true_relative_t%02d_mm" % EPOCH].values.astype(float)
deformation_est = df["deformation_est_t%02d_mm" % EPOCH].values.astype(float)

# 相对于参考点的真实高程误差和估计高程误差
height_true = df["delta_h_true_relative_m"].values.astype(float)
height_est = df["h_est_m_relative"].values.astype(float)

# ============================================================
# 获取参考点
# ============================================================
ref_idx = int(df["ref_idx"].iloc[0])

if ref_idx < 0 or ref_idx >= len(df):
    raise ValueError("ref_idx=%d 超出数据范围 0~%d" % (ref_idx, len(df) - 1))

ref_lon = longitude[ref_idx]
ref_lat = latitude[ref_idx]

print("PS 点数量       : %d" % len(df))
print("参考点索引      : %d" % ref_idx)
print("参考点 longitude: %.8f" % ref_lon)
print("参考点 latitude : %.8f" % ref_lat)

print("\n参考点第 %d 相真实形变 : %.6f mm" % (EPOCH, deformation_true[ref_idx]))
print("参考点第 %d 相估计形变 : %.6f mm" % (EPOCH, deformation_est[ref_idx]))
print("参考点真实高程误差     : %.6f m" % height_true[ref_idx])
print("参考点估计高程误差     : %.6f m" % height_est[ref_idx])

# ============================================================
# 查找三个目标位置最近的实际 PS 点
# ============================================================
selected_points = []

print("\n================ 选取的三个 PS 点 ================")

for name, target_lon, target_lat in TARGET_POINTS:
    distance = np.sqrt((longitude - target_lon) ** 2 + (latitude - target_lat) ** 2)
    idx = np.argmin(distance)

    point_id = df.iloc[idx]["point_id"]
    point_lon = longitude[idx]
    point_lat = latitude[idx]

    selected_points.append((name, idx, point_id, point_lon, point_lat))

    print("%s:" % name)
    print("  point_id  : %s" % str(point_id))
    print("  index     : %d" % idx)
    print("  longitude : %.8f" % point_lon)
    print("  latitude  : %.8f" % point_lat)

# ============================================================
# 计算误差
# estimated - true
# ============================================================
deformation_error = deformation_est - deformation_true
height_error = height_est - height_true

deformation_mae = np.mean(np.abs(deformation_error))
deformation_rmse = np.sqrt(np.mean(deformation_error ** 2))

height_mae = np.mean(np.abs(height_error))
height_rmse = np.sqrt(np.mean(height_error ** 2))

print("\n================ 精度评价 ================")
print("第 %d 相形变 MAE  : %.6f mm" % (EPOCH, deformation_mae))
print("第 %d 相形变 RMSE : %.6f mm" % (EPOCH, deformation_rmse))
print("高程误差 MAE      : %.6f m" % height_mae)
print("高程误差 RMSE     : %.6f m" % height_rmse)

print("\n================ 三个点的第80相形变误差 ================")

for name, idx, point_id, point_lon, point_lat in selected_points:
    print("%s | point_id=%s | error=%.6f mm" % (name, str(point_id), deformation_error[idx]))

# ============================================================
# 绘图范围
# ============================================================
lon_min, lon_max = longitude.min(), longitude.max()
lat_min, lat_max = latitude.min(), latitude.max()

deformation_vmin = -50
deformation_vmax = 50

height_limit = max(np.nanmax(np.abs(height_est)), 1.0)
deformation_error_limit = max(np.nanmax(np.abs(deformation_error)), 0.1)
height_error_limit = max(np.nanmax(np.abs(height_error)), 0.1)

# ============================================================
# 通用二维绘图函数
# ============================================================
def plot_map(values, filename, title, cbar_label, vmin, vmax, mark_reference=False, mark_selected=False):
    plt.figure(figsize=(7, 6))

    sc = plt.scatter(longitude, latitude, c=values, s=POINT_SIZE, cmap="jet", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc)
    cbar.set_label(cbar_label)

    # 标记参考点
    if mark_reference:
        plt.scatter(ref_lon, ref_lat, s=REF_SIZE, marker="^", facecolors="black", edgecolors="black", linewidths=0.5, zorder=10, label="Reference point")
        plt.legend(loc="lower right", frameon=True)

    # 标记 P1、P2、P3
    if mark_selected:
        for name, idx, point_id, point_lon, point_lat in selected_points:
            plt.scatter(point_lon, point_lat, s=SELECTED_POINT_SIZE, marker="o", facecolors="none", edgecolors="black", linewidths=1.5, zorder=12)
            plt.text(point_lon + 0.002, point_lat + 0.002, name, fontsize=10, fontweight="bold", color="black", zorder=13)

    plt.xlabel("Longitude / degree")
    plt.ylabel("Latitude / degree")
    plt.title(title)

    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / filename), dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 1. 第80相估计形变
# ============================================================
plot_map(
    deformation_est,
    "01_deformation_est_epoch80.png",
    "Estimated Deformation - Epoch %d" % EPOCH,
    "Cumulative deformation / mm",
    deformation_vmin,
    deformation_vmax,
    mark_reference=True
)

# ============================================================
# 2. 估计高程误差
# ============================================================
plot_map(
    height_est,
    "02_height_error_est.png",
    "Estimated Height Error",
    "Height error / m",
    -height_limit,
    height_limit,
    mark_reference=True
)

# ============================================================
# 3. 第80相形变误差
# 在该图中标记 P1、P2、P3
# ============================================================
plot_map(
    deformation_error,
    "03_deformation_error_epoch80.png",
    "Deformation Error - Epoch %d" % EPOCH,
    "Deformation error / mm",
    -deformation_error_limit,
    deformation_error_limit,
    mark_selected=True
)

# ============================================================
# 4. 高程误差差值
# ============================================================
plot_map(
    height_error,
    "04_height_error_difference.png",
    "Height Error Difference",
    "Height error difference / m",
    -height_error_limit,
    height_error_limit
)

print("\n四幅二维图已经保存到: %s" % OUTPUT_DIR)