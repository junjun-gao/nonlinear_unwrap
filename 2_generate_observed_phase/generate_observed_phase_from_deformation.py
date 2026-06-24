import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. 输入输出设置
# ============================================================

input_csv = "/data/test/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/1_generate_deformation/simulated_deformation_points.csv"

output_dir = Path("observed_phase_csv")
output_dir.mkdir(exist_ok=True)


# ============================================================
# 2. InSAR 仿真参数
# ============================================================

np.random.seed(42)

# 雷达波长，单位 m
wavelength = 0.056

# 时间间隔，单位 day
time_interval_days = 12

# 入射角，单位 degree
incidence_angle_deg = 35.0
incidence_angle_rad = np.deg2rad(incidence_angle_deg)

# 斜距，单位 m
slant_range = 800_000.0

# 垂直基线范围，单位 m
bperp_min = -150.0
bperp_max = 150.0

# 高程误差范围，单位 m
delta_h_min = -30.0
delta_h_max = 30.0

# 相位噪声标准差，单位 rad
phase_noise_std_list = [0.0, 0.2, 0.5, 0.8, 1.0]


# ============================================================
# 3. 相位缠绕函数
# ============================================================

def wrap_phase(phase):
    """
    将相位缠绕到 (-pi, pi]。
    """
    return np.angle(np.exp(1j * phase))


# ============================================================
# 4. 读取形变 CSV
# ============================================================

df = pd.read_csv(input_csv)

# 找到形变时间序列列
deformation_cols = [
    col for col in df.columns
    if col.startswith("deformation_t") and col.endswith("_mm")
]

# 按 deformation_t01_mm, deformation_t02_mm, ... 排序
deformation_cols = sorted(
    deformation_cols,
    key=lambda x: int(x.split("_t")[1].split("_mm")[0])
)

# 直接使用 CSV 里的时间点数量，比如 60
n_images = len(deformation_cols)
n_points = len(df)

print(f"Read CSV: {input_csv}")
print(f"Number of points: {n_points}")
print(f"Number of deformation time points: {n_images}")


# ============================================================
# 5. 直接读取 60 期形变，不进行插值
# ============================================================

# 形变，单位 mm
deformation_mm = df[deformation_cols].values

# mm 转 m
deformation_m = deformation_mm / 1000.0

# 时间轴
time_day = np.arange(n_images) * time_interval_days
time_year = time_day / 365.25


# ============================================================
# 6. 生成垂直基线和高程误差
# ============================================================

# 每个时间点对应一个垂直基线，单位 m
bperp = np.random.uniform(
    bperp_min,
    bperp_max,
    size=n_images
)

# 第 1 个时间点作为参考，基线设为 0
bperp[0] = 0.0

# 每个 PS 点随机生成一个高程误差，单位 m
delta_h = np.random.uniform(
    delta_h_min,
    delta_h_max,
    size=n_points
)


# ============================================================
# 7. 计算形变相位和高程误差相位
# ============================================================

# 形变相位
# phi_def = 4*pi/lambda * d
phase_deformation = (4.0 * np.pi / wavelength) * deformation_m

# 高程误差相位
# phi_h = 4*pi/lambda * B_perp/(r*sin(theta)) * delta_h
height_phase_factor = (
    4.0 * np.pi / wavelength
    * bperp / (slant_range * np.sin(incidence_angle_rad))
)

phase_height = delta_h[:, None] * height_phase_factor[None, :]

# 无噪声总相位
phase_total_clean = phase_deformation + phase_height


# ============================================================
# 8. 保存公共基线参数
# ============================================================

baseline_df = pd.DataFrame({
    "image_id": np.arange(1, n_images + 1),
    "time_day": time_day,
    "time_year": time_year,
    "bperp_m": bperp
})

baseline_csv = output_dir / f"simulation_baseline_{n_images}_images.csv"
baseline_df.to_csv(baseline_csv, index=False, encoding="utf-8-sig")

print(f"Baseline CSV saved: {baseline_csv}")


# ============================================================
# 9. 不同相位噪声下分别生成观测相位 CSV
# ============================================================

for phase_noise_std in phase_noise_std_list:

    # 相位噪声，单位 rad
    phase_noise = np.random.normal(
        loc=0.0,
        scale=phase_noise_std,
        size=phase_total_clean.shape
    )

    # 加噪后的未缠绕总相位
    phase_total_noisy = phase_total_clean + phase_noise

    # 缠绕观测相位，单位 rad
    observed_wrapped_phase = wrap_phase(phase_total_noisy)

    # 构建输出 DataFrame
    out_df = pd.DataFrame()

    # 保留点的基础信息
    for col in ["point_id", "type", "x", "y"]:
        if col in df.columns:
            out_df[col] = df[col]

    # 保存真实高程误差
    out_df["delta_h_true_m"] = delta_h

    # 保存最终形变量
    out_df["final_deformation_mm"] = deformation_mm[:, -1]

    # 保存 60 期形变真值
    for k in range(n_images):
        out_df[f"deformation_t{k+1:02d}_mm"] = deformation_mm[:, k]

    # 保存高程误差相位
    for k in range(n_images):
        out_df[f"height_phase_t{k+1:02d}_rad"] = phase_height[:, k]

    # 保存形变相位
    for k in range(n_images):
        out_df[f"deformation_phase_t{k+1:02d}_rad"] = phase_deformation[:, k]

    # 保存加噪后的未缠绕总相位
    for k in range(n_images):
        out_df[f"total_phase_unwrapped_t{k+1:02d}_rad"] = phase_total_noisy[:, k]

    # 保存最终观测缠绕相位
    for k in range(n_images):
        out_df[f"observed_phase_t{k+1:02d}_rad"] = observed_wrapped_phase[:, k]

    # 保存文件
    noise_tag = str(phase_noise_std).replace(".", "p")
    output_csv = output_dir / f"observed_phase_noise_{noise_tag}rad.csv"

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Observed phase CSV saved: {output_csv}")


print("All observed phase CSV files generated.")