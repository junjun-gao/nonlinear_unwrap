import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


output_dir = Path("simulate_seasonal_deformation")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Basic parameters
# ============================================================

np.random.seed(42)

n_points = 10000
n_times = 80
interval_days = 12.0
season_period_days = 365.25
noise_std = 0.6

width = 512
height = 512

lon_min, lon_max = 116.00, 116.30
lat_min, lat_max = 39.80, 40.00

time_days = np.arange(n_times, dtype=float) * interval_days
time_years = time_days / season_period_days


# ============================================================
# 2. Generate random PS locations
# ============================================================

x = np.random.uniform(0.0, width, n_points)
y = np.random.uniform(0.0, height, n_points)

longitude = lon_min + x / width * (lon_max - lon_min)
latitude = lat_min + y / height * (lat_max - lat_min)

labels = np.full(n_points, "Seasonal", dtype=object)


# ============================================================
# 3. Generate spatially correlated random field
# ============================================================

def generate_spatial_field(x, y, width, height, grid_size=100, smooth_sigma=10.0, value_min=0.0, value_max=1.0):
    random_grid = np.random.normal(0.0, 1.0, (grid_size, grid_size))
    smooth_grid = gaussian_filter(random_grid, sigma=smooth_sigma, mode="reflect")
    smooth_grid = (smooth_grid - smooth_grid.min()) / (smooth_grid.max() - smooth_grid.min() + 1e-12)
    grid_y = np.linspace(0.0, height, grid_size)
    grid_x = np.linspace(0.0, width, grid_size)
    interpolator = RegularGridInterpolator((grid_y, grid_x), smooth_grid, bounds_error=False, fill_value=None)
    field = interpolator(np.column_stack((y, x)))
    return value_min + field * (value_max - value_min)


# ============================================================
# 4. Generate seasonal parameters
# ============================================================

# Seasonal amplitude: 5 ~ 50 mm
seasonal_amplitude_mm = generate_spatial_field(x, y, width, height, grid_size=100, smooth_sigma=10.0, value_min=5.0, value_max=50.0)
seasonal_amplitude_mm += np.random.normal(0.0, 1.0, n_points)
seasonal_amplitude_mm = np.clip(seasonal_amplitude_mm, 5.0, 50.0)

# Linear velocity: -2 ~ 2 mm/year
trend_season = generate_spatial_field(x, y, width, height, grid_size=100, smooth_sigma=12.0, value_min=-2.0, value_max=2.0)
trend_season += np.random.normal(0.0, 0.15, n_points)
trend_season = np.clip(trend_season, -2.0, 2.0)

# Initial seasonal phase: -pi ~ pi
phase = generate_spatial_field(x, y, width, height, grid_size=100, smooth_sigma=14.0, value_min=-np.pi, value_max=np.pi)
phase += np.random.normal(0.0, 0.08, n_points)
phase = (phase + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# 5. Generate seasonal deformation
#
# d(t) = v*t + A*sin(2*pi*t/T + phi)
#
# v   : mm/year
# t   : year
# A   : mm
# T   : 365.25 days
# phi : initial phase
# ============================================================

basis_season = np.sin(2.0 * np.pi * time_days[None, :] / season_period_days + phase[:, None])
deformation_true = trend_season[:, None] * time_years[None, :] + seasonal_amplitude_mm[:, None] * basis_season


# ============================================================
# 6. Limit maximum true deformation to ±50 mm
#
# 不直接 clip 时序，避免把正弦波峰切平
# ============================================================

max_abs_deformation = np.max(np.abs(deformation_true), axis=1)
scale = np.minimum(1.0, 50.0 / (max_abs_deformation + 1e-12))

deformation_true *= scale[:, None]
seasonal_amplitude_mm *= scale
trend_season *= scale


# ============================================================
# 7. Add observation noise
# ============================================================

noise = np.random.normal(0.0, noise_std, deformation_true.shape)
deformation = deformation_true + noise


# ============================================================
# 8. Print statistics
# ============================================================

print("\n================ Seasonal deformation simulation ================")
print("Number of PS points       :", n_points)
print("Number of epochs          :", n_times)
print("Sampling interval         :", interval_days, "days")
print("Seasonal period           :", season_period_days, "days")
print("Observation duration      : %.3f years" % time_years[-1])
print("Approx. seasonal cycles   : %.3f" % (time_days[-1] / season_period_days))
print("Noise std                 : %.3f mm" % noise_std)
print("Amplitude range           : %.3f ~ %.3f mm" % (seasonal_amplitude_mm.min(), seasonal_amplitude_mm.max()))
print("Linear velocity range     : %.3f ~ %.3f mm/year" % (trend_season.min(), trend_season.max()))
print("Initial true deformation  : %.3f ~ %.3f mm" % (deformation_true[:, 0].min(), deformation_true[:, 0].max()))
print("True deformation range    : %.3f ~ %.3f mm" % (deformation_true.min(), deformation_true.max()))
print("Observed deformation range: %.3f ~ %.3f mm" % (deformation.min(), deformation.max()))

# ============================================================
# 9. Plot deformation maps at three epochs
# ============================================================

time_indices = [0, 39, 79]

for idx in time_indices:
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(longitude, latitude, c=deformation[:, idx], s=5, cmap="jet", vmin=-50, vmax=50)
    cbar = plt.colorbar(sc)
    cbar.set_label("Cumulative deformation / mm")
    plt.xlabel("Longitude / degree")
    plt.ylabel("Latitude / degree")
    plt.title("Seasonal Deformation - Epoch %d (%d days)" % (idx + 1, int(time_days[idx])))
    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.tight_layout()
    plt.savefig(str(output_dir / ("seasonal_deformation_time_%02d.png" % (idx + 1))), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 10. Randomly select one PS point
# ============================================================

random_ps_idx = np.random.randint(0, n_points)

print("\n================ Random PS point ================")
print("Point ID             :", random_ps_idx)
print("Longitude            : %.6f" % longitude[random_ps_idx])
print("Latitude             : %.6f" % latitude[random_ps_idx])
print("Seasonal amplitude   : %.3f mm" % seasonal_amplitude_mm[random_ps_idx])
print("Initial phase        : %.3f rad" % phase[random_ps_idx])
print("Linear velocity      : %.3f mm/year" % trend_season[random_ps_idx])
print("Initial deformation  : %.3f mm" % deformation_true[random_ps_idx, 0])
print("Maximum deformation  : %.3f mm" % np.max(deformation_true[random_ps_idx, :]))
print("Minimum deformation  : %.3f mm" % np.min(deformation_true[random_ps_idx, :]))


# ============================================================
# 11. Plot deformation time series of random PS point
# ============================================================

plt.figure(figsize=(9, 5))
plt.scatter(time_days, deformation_true[random_ps_idx, :], s=28, marker="o", label="True deformation")
plt.scatter(time_days, deformation[random_ps_idx, :], s=18, marker="x", label="Observed deformation")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("Time / days")
plt.ylabel("Deformation / mm")
plt.title("Seasonal Deformation of PS Point %d" % random_ps_idx)
plt.legend()
plt.tight_layout()
plt.savefig(str(output_dir / "random_ps_deformation_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 12. Save CSV
# ============================================================

data = {}
data["point_id"] = np.arange(n_points)
data["x"] = x
data["y"] = y
data["longitude"] = longitude
data["latitude"] = latitude
data["type"] = labels
data["seasonal_amplitude_mm"] = seasonal_amplitude_mm
data["seasonal_phase_rad"] = phase
data["linear_trend_mm_per_year"] = trend_season
data["initial_deformation_true_mm"] = deformation_true[:, 0]
data["final_deformation_mm"] = deformation[:, -1]

for k in range(n_times):
    data["deformation_t%02d_mm" % (k + 1)] = deformation[:, k]

df = pd.DataFrame(data)

csv_path = output_dir / "simulated_seasonal_deformation_points.csv"
df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")

print("\nCSV saved:", str(csv_path))
print("Random PS figure saved:", str(output_dir / "random_ps_deformation_timeseries.png"))
print(df.head())