import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, diags

# ============================================================
# 参数
# ============================================================
N_PS = 10000
N_EPOCHS = 80
WAVELENGTH = 0.0555
MAX_STEP_MM = WAVELENGTH * 1000.0 / 4.0 * 0.95
SEED = 2026
SPATIAL_SMOOTH_TIMES = 8
OUTPUT_DIR = Path("model_free_deformation_simulation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LON_MIN, LON_MAX = 116.00, 116.25
LAT_MIN, LAT_MAX = 39.85, 40.05

rng = np.random.RandomState(SEED)

# ============================================================
# 1. 生成 PS 点
# ============================================================
longitude = rng.uniform(LON_MIN, LON_MAX, N_PS)
latitude = rng.uniform(LAT_MIN, LAT_MAX, N_PS)
points = np.column_stack((longitude, latitude))

# ============================================================
# 2. Delaunay 建网
# ============================================================
tri = Delaunay(points)
simplices = tri.simplices
arcs = np.vstack((simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]))
arcs = np.sort(arcs, axis=1)
arcs = np.unique(arcs, axis=0)

row = np.r_[arcs[:, 0], arcs[:, 1], np.arange(N_PS)]
col = np.r_[arcs[:, 1], arcs[:, 0], np.arange(N_PS)]
A = csr_matrix((np.ones(len(row)), (row, col)), shape=(N_PS, N_PS))
degree = np.asarray(A.sum(axis=1)).ravel()
W = diags(1.0 / degree).dot(A)

print("PS points :", N_PS)
print("Arcs      :", len(arcs))
print("Epochs    :", N_EPOCHS)
print("lambda    : {:.4f} m".format(WAVELENGTH))
print("lambda / 4: {:.4f} mm".format(WAVELENGTH * 1000.0 / 4.0))
print("limit     : {:.4f} mm".format(MAX_STEP_MM))

# ============================================================
# 3. 生成空间相关随机场
# ============================================================
def generate_spatial_field():
    field = rng.normal(0.0, 1.0, N_PS)
    for _ in range(SPATIAL_SMOOTH_TIMES): field = 0.30 * field + 0.70 * W.dot(field)
    field -= np.mean(field)
    std = np.std(field)
    if std > 0: field /= std
    return field

# ============================================================
# 4. 生成非参数化时序形变
#
# 不使用：
# d(t) = vt
# d(t) = vt + A sin(...)
# d(t) = vt + at^2
# 等固定解析模型。
#
# 每个时相由新的空间相关随机场驱动，同时继承上一时相状态，
# 因而同时具有：
#   1) 空间相关性
#   2) 时间连续性
#   3) 非周期性
#   4) 非固定趋势
#   5) 不规则加速/减速
# ============================================================
deformation = np.zeros((N_PS, N_EPOCHS + 1), dtype=float)

state = generate_spatial_field()
previous_step = np.zeros(N_PS, dtype=float)

for epoch in range(1, N_EPOCHS + 1):
    innovation = generate_spatial_field()

    # 每个时相随机改变时间记忆强度，使时序不是固定 AR 模型
    rho = rng.uniform(0.72, 0.95)
    state = rho * state + np.sqrt(1.0 - rho ** 2) * innovation

    # 偶尔改变演化状态，形成不规则的加速、减速和方向变化
    if rng.rand() < 0.10:
        transition = generate_spatial_field()
        state = 0.60 * state + 0.40 * transition

    # 每个时相形变强度随机变化，不设置固定周期
    amplitude = rng.uniform(1.5, 5.0)
    step = amplitude * np.tanh(state)

    # 保持相邻时相连续，而不是完全独立的随机噪声
    step = 0.55 * previous_step + 0.45 * step

    # --------------------------------------------------------
    # 约束 1：单个 PS 相邻时相形变产生的相位差 < pi
    # --------------------------------------------------------
    max_point_step = np.max(np.abs(step))
    if max_point_step > MAX_STEP_MM: step *= MAX_STEP_MM / max_point_step

    # --------------------------------------------------------
    # 约束 2：额外限制 ARC 两端的差分形变增量 < lambda / 4
    # 这对后续 ARC 时间域解缠更友好
    # --------------------------------------------------------
    arc_step = step[arcs[:, 1]] - step[arcs[:, 0]]
    max_arc_step = np.max(np.abs(arc_step))
    if max_arc_step > MAX_STEP_MM: step *= MAX_STEP_MM / max_arc_step

    deformation[:, epoch] = deformation[:, epoch - 1] + step
    previous_step = step.copy()

# ============================================================
# 5. 检查相位约束
# ============================================================
temporal_increment = np.diff(deformation, axis=1)
max_temporal_step = np.max(np.abs(temporal_increment))
max_temporal_phase = 4.0 * np.pi * (max_temporal_step / 1000.0) / WAVELENGTH

max_arc_step = 0.0
for epoch in range(N_EPOCHS):
    step = temporal_increment[:, epoch]
    arc_step = step[arcs[:, 1]] - step[arcs[:, 0]]
    max_arc_step = max(max_arc_step, np.max(np.abs(arc_step)))

max_arc_phase = 4.0 * np.pi * (max_arc_step / 1000.0) / WAVELENGTH

print("\n================ Phase constraint ================")
print("Maximum PS temporal increment : {:.4f} mm".format(max_temporal_step))
print("Maximum PS phase increment    : {:.4f} rad".format(max_temporal_phase))
print("Maximum ARC temporal increment: {:.4f} mm".format(max_arc_step))
print("Maximum ARC phase increment   : {:.4f} rad".format(max_arc_phase))
print("pi                            : {:.4f} rad".format(np.pi))

# ============================================================
# 6. 检查相邻 PS 的空间相关性
# ============================================================
def arc_correlation(values):
    return np.corrcoef(values[arcs[:, 0]], values[arcs[:, 1]])[0, 1]

print("\n================ Spatial correlation ================")
print("Epoch 40 ARC correlation: {:.4f}".format(arc_correlation(deformation[:, 40])))
print("Epoch 80 ARC correlation: {:.4f}".format(arc_correlation(deformation[:, 80])))

# ============================================================
# 7. 保存形变数据
# ============================================================

# 归一化二维空间坐标
x = (longitude - LON_MIN) / (LON_MAX - LON_MIN)
y = (latitude - LAT_MIN) / (LAT_MAX - LAT_MIN)

output = {}

# ------------------------------------------------------------
# 80 个时相形变量：t01 ~ t80
# deformation[:, 0] 是初始 t0，不单独写入 deformation_t00_mm
# ------------------------------------------------------------
for epoch in range(1, N_EPOCHS + 1):
    output["deformation_t{:02d}_mm".format(epoch)] = deformation[:, epoch]

# ------------------------------------------------------------
# 与之前仿真 CSV 保持一致的附加字段
# ------------------------------------------------------------
output["final_deformation_mm"] = deformation[:, 80]
output["initial_deformation_true_mm"] = deformation[:, 0]
output["latitude"] = latitude

# Model-Free 形变不存在显式线性趋势和季节参数，因此设为 0
output["linear_trend_mm_per_year"] = np.zeros(N_PS)
output["longitude"] = longitude
output["point_id"] = np.arange(N_PS)
output["seasonal_amplitude_mm"] = np.zeros(N_PS)
output["seasonal_phase_rad"] = np.zeros(N_PS)
output["type"] = np.array(["ModelFree"] * N_PS)
output["x"] = x
output["y"] = y

df = pd.DataFrame(output)

csv_file = OUTPUT_DIR / "model_free_deformation.csv"
df.to_csv(str(csv_file), index=False)

# ------------------------------------------------------------
# Delaunay ARC 单独保存
# ------------------------------------------------------------
arc_df = pd.DataFrame({
    "arc_id": np.arange(len(arcs)),
    "point_1": arcs[:, 0],
    "point_2": arcs[:, 1]
})

arc_df.to_csv(str(OUTPUT_DIR / "delaunay_arcs.csv"), index=False)

print("\nDeformation CSV:", csv_file)
print("CSV shape      :", df.shape)
print("Columns        :", len(df.columns))

# ============================================================
# 8. 绘制 t0、t40、t80 形变量
# ============================================================

plot_epochs = [0, 40, 80]

DEFORMATION_VMIN = -150
DEFORMATION_VMAX = 150
cbar_ticks = np.arange(-150, 151, 50)

for epoch in plot_epochs:
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(
        longitude,
        latitude,
        c=deformation[:, epoch],
        s=5,
        cmap="jet",
        vmin=DEFORMATION_VMIN,
        vmax=DEFORMATION_VMAX
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("Cumulative deformation / mm")
    cbar.set_ticks(cbar_ticks)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Model-free deformation at epoch {}".format(epoch))

    plt.xlim(LON_MIN, LON_MAX)
    plt.ylim(LAT_MIN, LAT_MAX)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "deformation_epoch_{:02d}.png".format(epoch)), dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 9. 随机选择一个 PS，检查其非规则时间序列
# ============================================================
point_id = rng.randint(0, N_PS)

plt.figure(figsize=(9, 5))
plt.scatter(np.arange(N_EPOCHS + 1), deformation[point_id], s=20)
plt.xlabel("Epoch")
plt.ylabel("Cumulative deformation / mm")
plt.title("Temporal deformation of PS {}".format(point_id))
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "example_temporal_deformation.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nExample PS :", point_id)
print("Output     :", OUTPUT_DIR)