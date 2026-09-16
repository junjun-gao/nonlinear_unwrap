import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. Input and output
# ============================================================

input_csv = "/data/tests/junjun/nonlinear_unwrap/simulation/bspline_sim_seasonal/simulate_quadratic_deformation/simulated_quadratic_deformation_points.csv"

output_dir = Path("observed_phase_csv")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Sentinel-1-like simulation parameters
# ============================================================

np.random.seed(42)

wavelength = 0.0555
time_interval_days = 12.0

incidence_angle_deg = 35.0
incidence_angle_rad = np.deg2rad(incidence_angle_deg)

slant_range = 800000.0

bperp_min, bperp_max = -150.0, 150.0
delta_h_min, delta_h_max = -20.0, 20.0

phase_noise_std_list = [0.2, 0.5, 0.8]


# ============================================================
# 3. Phase wrapping function
# ============================================================

def wrap_phase(phase):
    return np.angle(np.exp(1j * phase))


# ============================================================
# 4. Read seasonal deformation CSV
# ============================================================

df = pd.read_csv(input_csv)

deformation_cols = [col for col in df.columns if col.startswith("deformation_t") and col.endswith("_mm")]
deformation_cols = sorted(deformation_cols, key=lambda x: int(x.split("_t")[1].split("_mm")[0]))

if len(deformation_cols) == 0:
    raise ValueError("No deformation columns such as deformation_t01_mm were found.")

n_points = len(df)
n_images = len(deformation_cols)

print("\n================ Observed phase simulation ================")
print("Input CSV             :", input_csv)
print("Number of PS points   :", n_points)
print("Number of acquisitions:", n_images)
print("Sampling interval     :", time_interval_days, "days")
print("Phase noise std list  :", phase_noise_std_list)


# ============================================================
# 5. Read absolute deformation time series
#
# 原始季节形变：
#
# d(t) = v*t + A*sin(2*pi*t/T + phi)
#
# 第 0 时刻允许不为 0
# ============================================================

deformation_absolute_mm = df[deformation_cols].values.astype(float)


# ============================================================
# 6. Convert absolute deformation to InSAR relative deformation
#
# InSAR 以第 0 景为参考：
#
# delta_d(t) = d(t) - d(t0)
#
# 因此：
#
# delta_d(t0) = 0
# ============================================================

deformation_relative_mm = deformation_absolute_mm - deformation_absolute_mm[:, [0]]
deformation_relative_m = deformation_relative_mm / 1000.0


# ============================================================
# 7. Time axis
# ============================================================

time_day = np.arange(n_images, dtype=float) * time_interval_days
time_year = time_day / 365.25

print("Observation duration   : %.3f days" % time_day[-1])
print("Observation duration   : %.3f years" % time_year[-1])


# ============================================================
# 8. Generate Sentinel-1-like perpendicular baselines
# ============================================================

bperp = np.random.uniform(bperp_min, bperp_max, n_images)
bperp[0] = 0.0


# ============================================================
# 9. Generate residual DEM errors
# ============================================================

delta_h = np.random.uniform(delta_h_min, delta_h_max, n_points)


# ============================================================
# 10. Calculate deformation phase
#
# phi_def = 4*pi/lambda * delta_d
#
# delta_d unit: meter
# ============================================================

phase_deformation = (4.0 * np.pi / wavelength) * deformation_relative_m


# ============================================================
# 11. Calculate residual topographic phase
#
# phi_h = 4*pi/lambda * Bperp/(R*sin(theta)) * delta_h
# ============================================================

height_phase_factor = (4.0 * np.pi / wavelength) * bperp / (slant_range * np.sin(incidence_angle_rad))
phase_height = delta_h[:, None] * height_phase_factor[None, :]


# ============================================================
# 12. Clean interferometric phase
# ============================================================

phase_total_clean = phase_deformation + phase_height

phase_deformation[:, 0] = 0.0
phase_height[:, 0] = 0.0
phase_total_clean[:, 0] = 0.0


# ============================================================
# 13. Print phase statistics
# ============================================================

print("\n================ Phase statistics ================")
print("Absolute deformation : %.3f ~ %.3f mm" % (deformation_absolute_mm.min(), deformation_absolute_mm.max()))
print("Relative deformation : %.3f ~ %.3f mm" % (deformation_relative_mm.min(), deformation_relative_mm.max()))
print("Residual height error: %.3f ~ %.3f m" % (delta_h.min(), delta_h.max()))
print("Bperp range          : %.3f ~ %.3f m" % (bperp.min(), bperp.max()))
print("Deformation phase    : %.3f ~ %.3f rad" % (phase_deformation.min(), phase_deformation.max()))
print("Height phase         : %.3f ~ %.3f rad" % (phase_height.min(), phase_height.max()))


# ============================================================
# 14. Save acquisition and baseline information
# ============================================================

baseline_df = pd.DataFrame({"image_id": np.arange(1, n_images + 1), "time_day": time_day, "time_year": time_year, "bperp_m": bperp})

baseline_df["wavelength_m"] = wavelength
baseline_df["incidence_angle_deg"] = incidence_angle_deg
baseline_df["slant_range_m"] = slant_range

baseline_csv = output_dir / ("simulation_baseline_%d_images.csv" % n_images)
baseline_df.to_csv(str(baseline_csv), index=False, encoding="utf-8-sig")

print("\nBaseline CSV saved:", str(baseline_csv))


# ============================================================
# 15. Generate observed phase for different noise levels
# ============================================================

for phase_noise_std in phase_noise_std_list:

    # --------------------------------------------------------
    # Add phase noise
    # --------------------------------------------------------

    phase_noise = np.random.normal(0.0, phase_noise_std, phase_total_clean.shape)
    phase_noise[:, 0] = 0.0

    phase_total_noisy = phase_total_clean + phase_noise
    observed_wrapped_phase = wrap_phase(phase_total_noisy)

    out_df = pd.DataFrame()

    # --------------------------------------------------------
    # Basic PS information
    # --------------------------------------------------------

    basic_cols = ["point_id", "type", "x", "y", "longitude", "latitude", "seasonal_amplitude_mm", "seasonal_phase_rad", "linear_trend_mm_per_year", "initial_deformation_true_mm"]

    for col in basic_cols:
        if col in df.columns:
            out_df[col] = df[col].values

    # --------------------------------------------------------
    # True simulation parameters
    # --------------------------------------------------------

    out_df["delta_h_true_m"] = delta_h
    out_df["phase_noise_std_rad"] = phase_noise_std

    # 相对最终形变量，供算法评价使用
    out_df["final_deformation_mm"] = deformation_relative_mm[:, -1]

    # 原始绝对最终形变量
    out_df["final_absolute_deformation_mm"] = deformation_absolute_mm[:, -1]

    # --------------------------------------------------------
    # Absolute deformation
    #
    # 保留原始季节模型真值
    # --------------------------------------------------------

    for k in range(n_images):
        out_df["absolute_deformation_t%02d_mm" % (k + 1)] = deformation_absolute_mm[:, k]

    # --------------------------------------------------------
    # Relative deformation
    #
    # 真正用于 InSAR 相位生成和算法精度评价
    # --------------------------------------------------------

    for k in range(n_images):
        out_df["deformation_t%02d_mm" % (k + 1)] = deformation_relative_mm[:, k]

    # --------------------------------------------------------
    # Residual topographic phase
    # --------------------------------------------------------

    for k in range(n_images):
        out_df["height_phase_t%02d_rad" % (k + 1)] = phase_height[:, k]

    # --------------------------------------------------------
    # Deformation phase
    # --------------------------------------------------------

    for k in range(n_images):
        out_df["deformation_phase_t%02d_rad" % (k + 1)] = phase_deformation[:, k]

    # --------------------------------------------------------
    # Phase noise
    # --------------------------------------------------------

    for k in range(n_images):
        out_df["phase_noise_t%02d_rad" % (k + 1)] = phase_noise[:, k]

    # --------------------------------------------------------
    # Unwrapped total phase
    # --------------------------------------------------------

    for k in range(n_images):
        out_df["total_phase_unwrapped_t%02d_rad" % (k + 1)] = phase_total_noisy[:, k]

    # --------------------------------------------------------
    # Final wrapped observed phase
    # --------------------------------------------------------

    for k in range(n_images):
        out_df["observed_phase_t%02d_rad" % (k + 1)] = observed_wrapped_phase[:, k]

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    noise_tag = str(phase_noise_std).replace(".", "p")
    output_csv = output_dir / ("observed_phase_noise_%srad.csv" % noise_tag)

    out_df.to_csv(str(output_csv), index=False, encoding="utf-8-sig")

    print("Observed phase saved:", str(output_csv))


print("\nAll observed phase simulations finished.")