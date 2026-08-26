#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# 1. Simulation parameters
# ============================================================

np.random.seed(42)

n_points = 10000
n_times = 80
interval_days = 12.0
year_days = 365.25
noise_std = 0.6

width = 512
height = 512

lon_min, lon_max = 116.00, 116.30
lat_min, lat_max = 39.80, 40.00

quadratic_min = -8.0
quadratic_max = -2.0
linear_min = -2.0
linear_max = 2.0

max_true_deformation_mm = 50.0

output_dir = Path("simulate_quadratic_deformation")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Spatially correlated random field
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
# 3. Generate PS coordinates
# ============================================================

x = np.random.uniform(0.0, width, n_points)
y = np.random.uniform(0.0, height, n_points)

longitude = lon_min + x / width * (lon_max - lon_min)
latitude = lat_min + y / height * (lat_max - lat_min)

labels = np.full(n_points, "Quadratic", dtype=object)


# ============================================================
# 4. Time axis
# ============================================================

time_days = np.arange(n_times, dtype=float) * interval_days
time_years = time_days / year_days

print("================ Quadratic deformation simulation ================")
print("Number of PS points   :", n_points)
print("Number of acquisitions:", n_times)
print("Sampling interval     : %.1f days" % interval_days)
print("Observation duration  : %.1f days" % time_days[-1])
print("Observation duration  : %.3f years" % time_years[-1])


# ============================================================
# 5. Generate spatially correlated quadratic parameters
#
# d(t) = v*t + a*t^2
#
# v : mm/year
# a : mm/year^2
# ============================================================

quadratic_coefficient = generate_spatial_field(
    x,
    y,
    width,
    height,
    grid_size=100,
    smooth_sigma=10.0,
    value_min=quadratic_min,
    value_max=quadratic_max,
)

quadratic_coefficient += np.random.normal(0.0, 0.15, n_points)
quadratic_coefficient = np.clip(quadratic_coefficient, quadratic_min, quadratic_max)

linear_trend = generate_spatial_field(
    x,
    y,
    width,
    height,
    grid_size=100,
    smooth_sigma=12.0,
    value_min=linear_min,
    value_max=linear_max,
)

linear_trend += np.random.normal(0.0, 0.15, n_points)
linear_trend = np.clip(linear_trend, linear_min, linear_max)


# ============================================================
# 6. Generate true quadratic deformation
#
# d(t) = v*t + a*t^2
# ============================================================

deformation_true = linear_trend[:, None] * time_years[None, :] + quadratic_coefficient[:, None] * time_years[None, :] ** 2


# ============================================================
# 7. Limit maximum true deformation to +/- 50 mm
#
# Scale the whole time series instead of clipping individual
# epochs, preserving the quadratic deformation shape.
# ============================================================

max_abs_deformation = np.max(np.abs(deformation_true), axis=1)
scale = np.minimum(1.0, max_true_deformation_mm / (max_abs_deformation + 1e-12))

deformation_true *= scale[:, None]
quadratic_coefficient *= scale
linear_trend *= scale


# ============================================================
# 8. Add deformation noise
# ============================================================

noise = np.random.normal(0.0, noise_std, deformation_true.shape)
deformation = deformation_true + noise


# ============================================================
# 9. Print deformation statistics
# ============================================================

print("\n================ Deformation statistics ================")
print("Quadratic coefficient : %.3f ~ %.3f mm/year^2" % (quadratic_coefficient.min(), quadratic_coefficient.max()))
print("Linear velocity       : %.3f ~ %.3f mm/year" % (linear_trend.min(), linear_trend.max()))
print("Initial true deformation: %.3f ~ %.3f mm" % (deformation_true[:, 0].min(), deformation_true[:, 0].max()))
print("True deformation range  : %.3f ~ %.3f mm" % (deformation_true.min(), deformation_true.max()))
print("Observed deformation range: %.3f ~ %.3f mm" % (deformation.min(), deformation.max()))
print("Noise std               : %.3f mm" % noise_std)


# ============================================================
# 10. Plot deformation fields
# ============================================================

time_indices = [0, 39, 79]

for idx in time_indices:
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(longitude, latitude, c=deformation[:, idx], s=5, cmap="jet", vmin=-50, vmax=50)
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Cumulative deformation / mm")
    plt.xlabel("Longitude / degree")
    plt.ylabel("Latitude / degree")
    plt.title("Quadratic Deformation - Epoch %d (%d days)" % (idx + 1, int(time_days[idx])))
    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.tight_layout()

    figure_path = output_dir / ("quadratic_deformation_time_%02d.png" % (idx + 1))
    plt.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close()

    print("Figure saved:", str(figure_path))


# ============================================================
# 11. Random PS deformation time series
# ============================================================

random_ps_idx = np.random.randint(0, n_points)

print("\n================ Random PS point ================")
print("Point ID              :", random_ps_idx)
print("Longitude             : %.6f" % longitude[random_ps_idx])
print("Latitude              : %.6f" % latitude[random_ps_idx])
print("Quadratic coefficient : %.3f mm/year^2" % quadratic_coefficient[random_ps_idx])
print("Linear velocity       : %.3f mm/year" % linear_trend[random_ps_idx])
print("Initial deformation   : %.3f mm" % deformation_true[random_ps_idx, 0])
print("Final deformation     : %.3f mm" % deformation_true[random_ps_idx, -1])
print("Maximum deformation   : %.3f mm" % np.max(deformation_true[random_ps_idx, :]))
print("Minimum deformation   : %.3f mm" % np.min(deformation_true[random_ps_idx, :]))

plt.figure(figsize=(9, 5))
plt.scatter(time_days, deformation_true[random_ps_idx, :], s=28, marker="o", label="True deformation")
plt.scatter(time_days, deformation[random_ps_idx, :], s=18, marker="x", label="Observed deformation")
plt.axhline(0.0, linewidth=0.8)
plt.xlabel("Time / days")
plt.ylabel("Deformation / mm")
plt.title("Quadratic Deformation of PS Point %d" % random_ps_idx)
plt.legend()
plt.tight_layout()

random_ps_figure = output_dir / "random_ps_deformation_timeseries.png"
plt.savefig(str(random_ps_figure), dpi=300, bbox_inches="tight")
plt.close()

print("Random PS figure saved:", str(random_ps_figure))


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
data["quadratic_coefficient_mm_per_year2"] = quadratic_coefficient
data["linear_trend_mm_per_year"] = linear_trend
data["initial_deformation_true_mm"] = deformation_true[:, 0]
data["final_deformation_mm"] = deformation[:, -1]

for k in range(n_times): data["deformation_t%02d_mm" % (k + 1)] = deformation[:, k]

df = pd.DataFrame(data)

csv_path = output_dir / "simulated_quadratic_deformation_points.csv"
df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")

print("\n================ Output ================")
print("CSV saved             :", str(csv_path))
print("Random PS figure      :", str(random_ps_figure))
print("Simulation finished.")