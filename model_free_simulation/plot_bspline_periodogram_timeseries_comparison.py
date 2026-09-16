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

BSPLINE_CSV = "/data/tests/junjun/nonlinear_unwrap/simulation/model_free_simulation/bspline_arc_unwrap_results/unwrapped_deformation_noise_0p2rad.csv"
LINEAR_PERIODOGRAM_CSV = "/data/tests/junjun/nonlinear_unwrap/simulation/model_free_simulation/linear_periodogram_results/unwrapped_deformation_noise_0p2rad.csv"
LINEAR_QUADRATIC_CSV = "/data/tests/junjun/nonlinear_unwrap/simulation/model_free_simulation/quadratic_periodogram_results/unwrapped_deformation_noise_0p2rad.csv"

OUTPUT_DIR = Path("algorithm_comparison_selected_points")
TIME_INTERVAL_DAYS = 12

# 固定三个 PS 点
SELECTED_POINTS = [
    ("P1", 3766),
    ("P2", 921),
    ("P3", 6060)
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 读取数据
# ============================================================

df_bspline = pd.read_csv(BSPLINE_CSV)
df_linear = pd.read_csv(LINEAR_PERIODOGRAM_CSV)
df_linear_quadratic = pd.read_csv(LINEAR_QUADRATIC_CSV)

for name, df in [("B-spline", df_bspline), ("Linear Periodogram", df_linear), ("Linear + Quadratic Periodogram", df_linear_quadratic)]:
    if "point_id" not in df.columns: raise ValueError("%s CSV 中缺少 point_id 列" % name)

if "longitude" not in df_bspline.columns or "latitude" not in df_bspline.columns: raise ValueError("B-spline CSV 中必须包含 longitude 和 latitude 列")

# ============================================================
# 检查固定点是否同时存在于三个结果中
# ============================================================

for name, point_id in SELECTED_POINTS:
    if not np.any(df_bspline["point_id"].values == point_id): raise ValueError("%s: point_id=%d 不存在于 B-spline CSV" % (name, point_id))
    if not np.any(df_linear["point_id"].values == point_id): raise ValueError("%s: point_id=%d 不存在于 Linear Periodogram CSV" % (name, point_id))
    if not np.any(df_linear_quadratic["point_id"].values == point_id): raise ValueError("%s: point_id=%d 不存在于 Linear + Quadratic Periodogram CSV" % (name, point_id))

# ============================================================
# 提取时相编号
# ============================================================

def get_epoch_number(col):
    match = re.search(r"_t(\d+)_mm$", col)
    if match is None: raise ValueError("无法从字段名中提取时相编号: %s" % col)
    return int(match.group(1))

# ============================================================
# 自动查找时序字段
# ============================================================

est_cols = [col for col in df_bspline.columns if re.match(r"^deformation_est_t\d+_mm$", col)]
true_cols = [col for col in df_bspline.columns if re.match(r"^deformation_true_relative_t\d+_mm$", col)]

est_cols = sorted(est_cols, key=get_epoch_number)
true_cols = sorted(true_cols, key=get_epoch_number)

if len(est_cols) == 0: raise ValueError("没有找到 deformation_est_tXX_mm 字段")
if len(true_cols) == 0: raise ValueError("没有找到 deformation_true_relative_tXX_mm 字段")
if len(est_cols) != len(true_cols): raise ValueError("估计形变和真实形变的时相数量不一致")

for col in est_cols:
    if col not in df_linear.columns: raise ValueError("Linear Periodogram CSV 中缺少字段: %s" % col)
    if col not in df_linear_quadratic.columns: raise ValueError("Linear + Quadratic Periodogram CSV 中缺少字段: %s" % col)

n_times = len(est_cols)
time_days = np.arange(n_times) * TIME_INTERVAL_DAYS

print("==========================================")
print("数据基本信息")
print("==========================================")
print("B-spline PS 数量                  : %d" % len(df_bspline))
print("Linear Periodogram PS 数量        : %d" % len(df_linear))
print("Linear + Quadratic PS 数量        : %d" % len(df_linear_quadratic))
print("时相数量                          : %d" % n_times)

# ============================================================
# 输出三个固定点的信息
# ============================================================

print("\n==========================================")
print("固定 PS 点")
print("==========================================")

for name, point_id in SELECTED_POINTS:
    row = df_bspline[df_bspline["point_id"] == point_id].iloc[0]
    longitude = float(row["longitude"])
    latitude = float(row["latitude"])

    print("%s:" % name)
    print("  point_id  : %d" % point_id)
    print("  longitude : %.8f" % longitude)
    print("  latitude  : %.8f" % latitude)

# ============================================================
# 分别绘制三个点
# ============================================================

for name, point_id in SELECTED_POINTS:
    row_bspline = df_bspline[df_bspline["point_id"] == point_id].iloc[0]
    row_linear = df_linear[df_linear["point_id"] == point_id].iloc[0]
    row_linear_quadratic = df_linear_quadratic[df_linear_quadratic["point_id"] == point_id].iloc[0]

    longitude = float(row_bspline["longitude"])
    latitude = float(row_bspline["latitude"])

    # ========================================================
    # 真实值和三种算法估计值
    # ========================================================

    deformation_true = row_bspline[true_cols].values.astype(float)
    deformation_bspline = row_bspline[est_cols].values.astype(float)
    deformation_linear = row_linear[est_cols].values.astype(float)
    deformation_linear_quadratic = row_linear_quadratic[est_cols].values.astype(float)

    # ========================================================
    # 精度统计
    # ========================================================

    error_bspline = deformation_bspline - deformation_true
    error_linear = deformation_linear - deformation_true
    error_linear_quadratic = deformation_linear_quadratic - deformation_true

    mae_bspline = np.mean(np.abs(error_bspline))
    rmse_bspline = np.sqrt(np.mean(error_bspline ** 2))
    mae_linear = np.mean(np.abs(error_linear))
    rmse_linear = np.sqrt(np.mean(error_linear ** 2))
    mae_linear_quadratic = np.mean(np.abs(error_linear_quadratic))
    rmse_linear_quadratic = np.sqrt(np.mean(error_linear_quadratic ** 2))

    print("\n==========================================")
    print("%s | point_id = %d" % (name, point_id))
    print("==========================================")
    print("longitude                    : %.8f" % longitude)
    print("latitude                     : %.8f" % latitude)
    print("B-spline MAE                 : %.4f mm" % mae_bspline)
    print("B-spline RMSE                : %.4f mm" % rmse_bspline)
    print("Linear Periodogram MAE       : %.4f mm" % mae_linear)
    print("Linear Periodogram RMSE      : %.4f mm" % rmse_linear)
    print("Linear + Quadratic MAE       : %.4f mm" % mae_linear_quadratic)
    print("Linear + Quadratic RMSE      : %.4f mm" % rmse_linear_quadratic)

    # ========================================================
    # 绘图
    # 长宽比 3:1
    # True                    : 实线
    # B-spline                : 方形点
    # Linear Periodogram      : 三角形点
    # Linear + Quadratic      : 圆形点
    # P3 图例放左下角
    # 标题显示经纬度
    # ========================================================

    plt.figure(figsize=(12, 4))

    plt.plot(time_days, deformation_true, linewidth=2.0, color="black", label="True")
    plt.scatter(time_days, deformation_bspline, s=30, marker="s", label="B-spline")
    plt.scatter(time_days, deformation_linear, s=34, marker="^", label="Linear Periodogram")
    plt.scatter(time_days, deformation_linear_quadratic, s=30, marker="o", label="Linear + Quadratic Periodogram")

    plt.xlabel("Time / day")
    plt.ylabel("Cumulative deformation / mm")
    plt.title("%s - Lon: %.6f, Lat: %.6f" % (name, longitude, latitude))

    if name == "P3": plt.legend(loc="lower left")
    else: plt.legend()

    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_file = OUTPUT_DIR / ("%s_point_%d_timeseries_comparison.png" % (name, point_id))
    plt.savefig(str(output_file), dpi=300, bbox_inches="tight")
    plt.close()

    print("图像已保存: %s" % output_file)

print("\n全部绘制完成。")