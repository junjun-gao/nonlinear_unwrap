import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from pathlib import Path

# ============================================================
# 参数
# ============================================================
BSPLINE_CSV = "/data/tests/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/bspline_sim_seasonal/bspline_seasonal_arc_unwrap_results/unwrapped_deformation_noise_0p2rad.csv"
PERIODOGRAM_CSV = "/data/tests/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/bspline_sim_seasonal/seasonal_periodogram_results/unwrapped_deformation_noise_0p2rad.csv"
OUTPUT_DIR = Path("algorithm_comparison_selected_points")
TIME_INTERVAL_DAYS = 12

# 三个红圈的大致中心位置
TARGET_POINTS = [
    ("P1", 116.0514, 39.9602),
    ("P2", 116.2057, 39.9271),
    ("P3", 116.0127, 39.8056)
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 读取数据
# ============================================================
df_bspline = pd.read_csv(BSPLINE_CSV)
df_periodogram = pd.read_csv(PERIODOGRAM_CSV)

if "point_id" not in df_bspline.columns or "point_id" not in df_periodogram.columns:
    raise ValueError("两个 CSV 中都必须包含 point_id 列")

if "longitude" not in df_bspline.columns or "latitude" not in df_bspline.columns:
    raise ValueError("B-spline CSV 中必须包含 longitude 和 latitude 列")

# ============================================================
# 找到两个 CSV 共同存在的 PS 点
# ============================================================
common_ids = np.intersect1d(df_bspline["point_id"].values, df_periodogram["point_id"].values)

if len(common_ids) == 0:
    raise ValueError("两个 CSV 中没有共同的 point_id")

df_common = df_bspline[df_bspline["point_id"].isin(common_ids)].copy().reset_index(drop=True)

# ============================================================
# 提取时相编号
# ============================================================
def get_epoch_number(col):
    match = re.search(r"_t(\d+)_mm$", col)
    if match is None:
        raise ValueError("无法从字段名中提取时相编号: %s" % col)
    return int(match.group(1))

# ============================================================
# 自动查找时序字段
# ============================================================
est_cols = [col for col in df_bspline.columns if re.match(r"^deformation_est_t\d+_mm$", col)]
true_cols = [col for col in df_bspline.columns if re.match(r"^deformation_true_relative_t\d+_mm$", col)]

est_cols = sorted(est_cols, key=get_epoch_number)
true_cols = sorted(true_cols, key=get_epoch_number)

if len(est_cols) == 0:
    raise ValueError("没有找到 deformation_est_tXX_mm 字段")

if len(true_cols) == 0:
    raise ValueError("没有找到 deformation_true_relative_tXX_mm 字段")

if len(est_cols) != len(true_cols):
    raise ValueError("估计形变和真实形变的时相数量不一致")

for col in est_cols:
    if col not in df_periodogram.columns:
        raise ValueError("Periodogram CSV 中缺少字段: %s" % col)

n_times = len(est_cols)
time_days = np.arange(n_times) * TIME_INTERVAL_DAYS

print("==========================================")
print("数据基本信息")
print("==========================================")
print("B-spline PS 数量    : %d" % len(df_bspline))
print("Periodogram PS 数量 : %d" % len(df_periodogram))
print("共同 PS 数量        : %d" % len(common_ids))
print("时相数量            : %d" % n_times)

# ============================================================
# 根据目标经纬度查找最近的 PS 点
# ============================================================
selected_points = []

for name, target_lon, target_lat in TARGET_POINTS:
    distance = np.sqrt((df_common["longitude"].values - target_lon) ** 2 + (df_common["latitude"].values - target_lat) ** 2)
    nearest_idx = np.argmin(distance)

    point_id = df_common.iloc[nearest_idx]["point_id"]
    longitude = float(df_common.iloc[nearest_idx]["longitude"])
    latitude = float(df_common.iloc[nearest_idx]["latitude"])

    selected_points.append((name, point_id, longitude, latitude))

    print("\n%s:" % name)
    print("  目标 longitude : %.8f" % target_lon)
    print("  目标 latitude  : %.8f" % target_lat)
    print("  point_id       : %s" % str(point_id))
    print("  实际 longitude : %.8f" % longitude)
    print("  实际 latitude  : %.8f" % latitude)
    print("  坐标距离       : %.8f" % distance[nearest_idx])

# ============================================================
# 分别绘制三个点
# ============================================================
for name, point_id, longitude, latitude in selected_points:
    row_bspline = df_bspline[df_bspline["point_id"] == point_id].iloc[0]
    row_periodogram = df_periodogram[df_periodogram["point_id"] == point_id].iloc[0]

    # 真实值和两种算法估计值
    deformation_true = row_bspline[true_cols].values.astype(float)
    deformation_bspline = row_bspline[est_cols].values.astype(float)
    deformation_periodogram = row_periodogram[est_cols].values.astype(float)

    # ========================================================
    # 精度统计
    # ========================================================
    error_bspline = deformation_bspline - deformation_true
    error_periodogram = deformation_periodogram - deformation_true

    mae_bspline = np.mean(np.abs(error_bspline))
    rmse_bspline = np.sqrt(np.mean(error_bspline ** 2))

    mae_periodogram = np.mean(np.abs(error_periodogram))
    rmse_periodogram = np.sqrt(np.mean(error_periodogram ** 2))

    print("\n==========================================")
    print("%s  point_id = %s" % (name, str(point_id)))
    print("==========================================")
    print("longitude        : %.8f" % longitude)
    print("latitude         : %.8f" % latitude)
    print("B-spline MAE     : %.4f mm" % mae_bspline)
    print("B-spline RMSE    : %.4f mm" % rmse_bspline)
    print("Periodogram MAE  : %.4f mm" % mae_periodogram)
    print("Periodogram RMSE : %.4f mm" % rmse_periodogram)

    # ========================================================
    # 绘图：只有点，不连线
    # ========================================================
    plt.figure(figsize=(9, 5.5))

    plt.scatter(time_days, deformation_true, s=28, marker="o", c="black", label="True")
    plt.scatter(time_days, deformation_bspline, s=30, marker="s", label="B-spline")
    plt.scatter(time_days, deformation_periodogram, s=34, marker="^", label="Periodogram")

    plt.xlabel("Time / day")
    plt.ylabel("Cumulative deformation / mm")
    plt.title("%s - Point %s" % (name, str(point_id)))

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_file = OUTPUT_DIR / ("%s_point_%s_timeseries_comparison.png" % (name, str(point_id)))
    plt.savefig(str(output_file), dpi=300, bbox_inches="tight")
    plt.close()

    print("图像已保存: %s" % output_file)

print("\n全部绘制完成。")